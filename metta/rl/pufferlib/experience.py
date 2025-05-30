"""
This file implements an Experience class for storing and managing experience data during reinforcement
learning training.

The Experience class provides:
- Flat tensor storage for observations, actions, rewards, etc.
- Array views for faster indexing
- Support for LSTM state storage
- Minibatch handling for training
- CPU/GPU memory management

Key features:
- Stores trajectories in fixed-size tensors
- Supports both CPU and GPU storage with optional CPU offloading
- Handles LSTM hidden states if using recurrent policies
- Provides numpy array views for efficient indexing
- Manages minibatch creation for training
"""

from typing import Optional, Tuple

import numpy as np
import pufferlib
import pufferlib.pytorch
import torch


class Experience:
    """Flat tensor storage and array views for faster indexing"""

    def __init__(
        self,
        batch_size: int,
        bptt_horizon: int,
        minibatch_size: Optional[int],
        hidden_size: int,
        obs_shape: Tuple[int, ...],
        obs_dtype: np.dtype,
        atn_shape: Tuple[int, ...],
        atn_dtype: np.dtype,
        cpu_offload: bool = False,
        device: str = "cuda",
        lstm: Optional[torch.nn.LSTM] = None,
        lstm_total_agents: int = 0,
    ):
        if minibatch_size is None:
            minibatch_size = batch_size

        # Convert numpy dtypes to torch dtypes
        obs_dtype_torch = pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_dtype]
        atn_dtype_torch = pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_dtype]
        pin = device == "cuda" and cpu_offload

        # Create tensors without using *args unpacking to make Pylance happy
        tensor_device = "cpu" if pin else device

        # Create a fully specified size tuple for each tensor
        obs_size = (batch_size,) + obs_shape  # Explicitly create a new tuple
        atn_size = (batch_size,) + atn_shape  # Explicitly create a new tuple

        # Create tensors with explicit size tuples
        self.obs = torch.zeros(size=obs_size, dtype=obs_dtype_torch, device=tensor_device)
        self.actions = torch.zeros(size=atn_size, dtype=atn_dtype_torch, device="cpu")
        self.logprobs = torch.zeros(size=(batch_size,), device="cpu")
        self.rewards = torch.zeros(size=(batch_size,), device="cpu")
        self.dones = torch.zeros(size=(batch_size,), device="cpu")
        self.truncateds = torch.zeros(size=(batch_size,), device="cpu")
        self.values = torch.zeros(size=(batch_size,), device="cpu")

        # Apply pin_memory if needed
        if pin:
            self.obs = self.obs.pin_memory()
            self.actions = self.actions.pin_memory()
            self.logprobs = self.logprobs.pin_memory()
            self.rewards = self.rewards.pin_memory()
            self.dones = self.dones.pin_memory()
            self.truncateds = self.truncateds.pin_memory()
            self.values = self.values.pin_memory()

        self.e3b_inv = 10 * torch.eye(hidden_size).repeat(lstm_total_agents, 1, 1).to(device)

        # Create numpy views with explicit types
        self.actions_np: np.ndarray = np.asarray(self.actions)
        self.logprobs_np: np.ndarray = np.asarray(self.logprobs)
        self.rewards_np: np.ndarray = np.asarray(self.rewards)
        self.dones_np: np.ndarray = np.asarray(self.dones)
        self.truncateds_np: np.ndarray = np.asarray(self.truncateds)
        self.values_np: np.ndarray = np.asarray(self.values)

        assert lstm is not None, "LSTM instance is required for experience buffer"
        assert lstm_total_agents > 0, f"lstm_total_agents must be positive, got {lstm_total_agents}"
        shape = (lstm.num_layers, lstm_total_agents, lstm.hidden_size)
        self.lstm_h: torch.Tensor = torch.zeros(shape).to(device, non_blocking=True)
        self.lstm_c: torch.Tensor = torch.zeros(shape).to(device, non_blocking=True)

        num_minibatches = batch_size / minibatch_size
        self.num_minibatches: int = int(num_minibatches)
        if self.num_minibatches != num_minibatches:
            raise ValueError(f"batch_size {batch_size} must be divisible by minibatch_size {minibatch_size}")

        minibatch_rows = minibatch_size / bptt_horizon
        self.minibatch_rows: int = int(minibatch_rows)
        if self.minibatch_rows != minibatch_rows:
            raise ValueError(f"minibatch_size {minibatch_size} must be divisible by bptt_horizon {bptt_horizon}")

        # Store parameters
        self.batch_size: int = batch_size
        self.bptt_horizon: int = bptt_horizon
        self.minibatch_size: int = minibatch_size
        self.device: str = device

        # Initialize sort keys
        self.sort_keys: np.ndarray = np.zeros((batch_size, 3), dtype=np.int32)
        self.sort_keys[:, 0] = np.arange(batch_size)
        self.ptr: int = 0
        self.step: int = 0

        # Batch indices will be set by sort_training_data
        self.b_idxs_obs: Optional[torch.Tensor] = None
        self.b_idxs: Optional[torch.Tensor] = None
        self.b_idxs_flat: Optional[torch.Tensor] = None
        self.b_actions: Optional[torch.Tensor] = None
        self.b_logprobs: Optional[torch.Tensor] = None
        self.b_dones: Optional[torch.Tensor] = None
        self.b_values: Optional[torch.Tensor] = None
        self.b_advantages: Optional[torch.Tensor] = None
        self.b_obs: Optional[torch.Tensor] = None
        self.b_returns: Optional[torch.Tensor] = None
        self.returns_np: Optional[np.ndarray] = None

    @property
    def full(self) -> bool:
        return self.ptr >= self.batch_size

    def store(
        self,
        obs: torch.Tensor,
        value: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        env_id: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        assert not isinstance(env_id, slice), (
            f"TypeError: env_id expected to be a tensor, not a slice. Got {type(env_id).__name__} instead. "
        )

        # Get current pointer and calculate indices
        ptr = self.ptr

        mask_np: np.ndarray = mask.cpu().numpy()
        indices = np.where(mask_np)[0]

        # Calculate how many indices we can actually store
        remaining_space = self.batch_size - ptr
        num_indices_to_store = min(indices.size, remaining_space)

        # TODO -- we constantly overrun the buffer and truncate; typically we have 1920 indices for 1024 slots

        end = ptr + num_indices_to_store
        dst = slice(ptr, end)

        cpu_inds = indices[:num_indices_to_store]

        self.obs[dst] = obs.to(self.obs.device, non_blocking=True)[cpu_inds]
        self.values_np[dst] = value.cpu().numpy()[cpu_inds]
        self.actions_np[dst] = action.cpu().numpy()[cpu_inds]
        self.logprobs_np[dst] = logprob.cpu().numpy()[cpu_inds]
        self.rewards_np[dst] = reward.cpu().numpy()[cpu_inds]
        self.dones_np[dst] = done.cpu().numpy()[cpu_inds]

        self.sort_keys[dst, 1] = env_id.cpu().numpy()[cpu_inds]
        self.sort_keys[dst, 2] = self.step

        # Update pointer and step
        self.ptr = end
        self.step += 1

    def sort_training_data(self) -> np.ndarray:
        idxs: np.ndarray = np.lexsort((self.sort_keys[:, 2], self.sort_keys[:, 1]))
        self.b_idxs_obs = (
            torch.as_tensor(
                idxs.reshape(self.minibatch_rows, self.num_minibatches, self.bptt_horizon).transpose(1, 0, -1)
            )
            .to(self.obs.device)
            .long()
        )
        self.b_idxs = self.b_idxs_obs.to(self.device, non_blocking=True)
        self.b_idxs_flat = self.b_idxs.reshape(self.num_minibatches, self.minibatch_size)
        self.sort_keys[:, 1:] = 0
        return idxs

    def flatten_batch(self, advantages_np: np.ndarray) -> None:
        advantages: torch.Tensor = torch.as_tensor(advantages_np).to(self.device, non_blocking=True)

        if self.b_idxs_obs is None:
            raise ValueError("b_idxs_obs is None - call sort_training_data first")

        b_idxs, b_flat = self.b_idxs, self.b_idxs_flat

        # Process the batch data
        self.b_actions = self.actions.to(self.device, non_blocking=True, dtype=torch.long)
        self.b_logprobs = self.logprobs.to(self.device, non_blocking=True)
        self.b_dones = self.dones.to(self.device, non_blocking=True)
        self.b_values = self.values.to(self.device, non_blocking=True)

        # Reshape advantages for minibatches
        self.b_advantages = (
            advantages.reshape(self.minibatch_rows, self.num_minibatches, self.bptt_horizon)
            .transpose(0, 1)
            .reshape(self.num_minibatches, self.minibatch_size)
        )

        self.returns_np = advantages_np + self.values_np

        # Process the rest of the batch data
        self.b_obs = self.obs[self.b_idxs_obs]
        self.b_actions = self.b_actions[b_idxs].contiguous()
        self.b_logprobs = self.b_logprobs[b_idxs]
        self.b_dones = self.b_dones[b_idxs]
        self.b_values = self.b_values[b_flat]
        self.b_returns = self.b_advantages + self.b_values
