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

        obs_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_dtype]
        atn_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_dtype]
        pin = device == "cuda" and cpu_offload
        self.obs = torch.zeros(
            batch_size, *obs_shape, dtype=obs_dtype, pin_memory=pin, device=device if not pin else "cpu"
        )
        self.actions = torch.zeros(batch_size, *atn_shape, dtype=atn_dtype, pin_memory=pin)
        self.logprobs = torch.zeros(batch_size, pin_memory=pin)
        self.rewards = torch.zeros(batch_size, pin_memory=pin)
        self.dones = torch.zeros(batch_size, pin_memory=pin)
        self.truncateds = torch.zeros(batch_size, pin_memory=pin)
        self.values = torch.zeros(batch_size, pin_memory=pin)
        self.e3b_inv = 10 * torch.eye(hidden_size).repeat(lstm_total_agents, 1, 1).to(device)

        self.actions_np: np.ndarray = np.asarray(self.actions)
        self.logprobs_np: np.ndarray = np.asarray(self.logprobs)
        self.rewards_np: np.ndarray = np.asarray(self.rewards)
        self.dones_np: np.ndarray = np.asarray(self.dones)
        self.truncateds_np: np.ndarray = np.asarray(self.truncateds)
        self.values_np: np.ndarray = np.asarray(self.values)

        assert lstm is not None
        assert lstm_total_agents > 0
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
        mask_np: np.ndarray = mask.cpu().numpy()

        # Get current pointer and calculate indices
        ptr: int = self.ptr
        indices: np.ndarray = np.where(mask_np)[0]
        num_indices: int = indices.size
        end: int = ptr + num_indices
        dst: slice = slice(ptr, end)

        # Zero-copy indexing for contiguous env_id
        if num_indices == mask.size and isinstance(env_id, slice):
            cpu_inds = slice(0, min(self.batch_size - ptr, num_indices))
        else:
            cpu_inds = indices[: self.batch_size - ptr]
            torch.as_tensor(indices).to(self.obs.device, non_blocking=True)

        self.obs[dst] = obs.to(self.obs.device, non_blocking=True)[cpu_inds]
        self.values_np[dst] = value.cpu().numpy()[cpu_inds]
        self.actions_np[dst] = action[cpu_inds]
        self.logprobs_np[dst] = logprob.cpu().numpy()[cpu_inds]
        self.rewards_np[dst] = reward.cpu().numpy()[cpu_inds]
        self.dones_np[dst] = done.cpu().numpy()[cpu_inds]
        if isinstance(env_id, slice):
            self.sort_keys[dst, 1] = np.arange(cpu_inds.start, cpu_inds.stop, dtype=np.int32)
        else:
            # Move env_id to CPU before indexing and converting to numpy
            if isinstance(env_id, torch.Tensor) and env_id.device.type == "cuda":
                self.sort_keys[dst, 1] = env_id.cpu()[cpu_inds].numpy()
            else:
                self.sort_keys[dst, 1] = env_id[cpu_inds]

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
        self.b_obs = self.obs[self.b_idxs_obs]
        self.b_actions = self.b_actions[b_idxs].contiguous()
        self.b_logprobs = self.b_logprobs[b_idxs]
        self.b_dones = self.b_dones[b_idxs]
        self.b_values = self.b_values[b_flat]
        self.b_returns = self.b_advantages + self.b_values
