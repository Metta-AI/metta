"""
This file implements an Experience class for storing and managing experience data during reinforcement
learning training.

The Experience class provides:
- Segmented tensor storage for observations, actions, rewards, etc.
- Support for BPTT (Backpropagation Through Time) with configurable horizon
- Prioritized experience replay with importance sampling
- LSTM state management for recurrent policies
- Zero-copy operations where possible
- Efficient minibatch creation for training

Key features:
- Stores trajectories in segmented tensors for BPTT
- Supports both CPU and GPU storage with optional CPU offloading
- Handles LSTM hidden states if using recurrent policies
- Provides prioritized sampling for training
- Manages minibatch creation for training
"""

from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor


class Experience:
    """Segmented tensor storage for RL experience with BPTT support."""

    def __init__(
        self,
        total_agents: int,
        batch_size: int,
        bptt_horizon: int,
        minibatch_size: int,
        max_minibatch_size: int,
        obs_space,
        atn_space,
        device: torch.device | str,
        hidden_size: int,
        cpu_offload: bool = False,
        num_lstm_layers: int = 2,
        agents_per_batch: Optional[int] = None,
        reset_lstm_state_between_episodes: bool = True,
    ):
        """Initialize experience buffer with segmented storage."""
        # Store parameters
        self.total_agents = total_agents
        self.batch_size: int = batch_size
        self.bptt_horizon: int = bptt_horizon
        self.reset_lstm_state_between_episodes = reset_lstm_state_between_episodes
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.cpu_offload = cpu_offload

        # Calculate segments
        self.segments = batch_size // bptt_horizon
        if total_agents > self.segments:
            mini_batch_size = total_agents * bptt_horizon
            raise ValueError(
                f"batch_size ({batch_size}) is too small for {total_agents} agents.\n"
                f"Segments = batch_size // bptt_horizon = {batch_size} // {bptt_horizon} = {self.segments}\n"
                f"But we need segments >= total_agents ({total_agents}).\n"
                f"Please set trainer.batch_size >= {mini_batch_size} in your configuration."
            )

        # Determine tensor device and dtype
        obs_device = "cpu" if cpu_offload else self.device
        obs_dtype = torch.float32 if obs_space.dtype == np.float32 else torch.uint8
        pin = str(self.device).startswith("cuda") and cpu_offload

        # Create segmented tensor storage
        self.obs = torch.zeros(
            self.segments,
            bptt_horizon,
            *obs_space.shape,
            dtype=obs_dtype,
            pin_memory=pin,
            device=obs_device,
        )

        # Action tensor with proper dtype
        atn_dtype = torch.int32 if np.issubdtype(atn_space.dtype, np.integer) else torch.float32
        self.actions = torch.zeros(self.segments, bptt_horizon, *atn_space.shape, device=self.device, dtype=atn_dtype)

        # Create value and policy tensors
        self.values = torch.zeros(self.segments, bptt_horizon, device=self.device)
        self.logprobs = torch.zeros(self.segments, bptt_horizon, device=self.device)
        self.rewards = torch.zeros(self.segments, bptt_horizon, device=self.device)
        self.dones = torch.zeros(self.segments, bptt_horizon, device=self.device)
        self.truncateds = torch.zeros(self.segments, bptt_horizon, device=self.device)
        self.ratio = torch.ones(self.segments, bptt_horizon, device=self.device)

        # Episode tracking
        self.ep_lengths = torch.zeros(total_agents, device=self.device, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, device=self.device, dtype=torch.int32) % self.segments
        self.free_idx = total_agents % self.segments

        # LSTM state management
        self.lstm_h: Dict[int, Tensor] = {}
        self.lstm_c: Dict[int, Tensor] = {}
        assert num_lstm_layers > 0, f"num_lstm_layers must be positive, got {num_lstm_layers}"
        assert hidden_size > 0, f"hidden_size must be positive, got {hidden_size}"

        # Use provided agents_per_batch or default to total_agents
        if agents_per_batch is None:
            agents_per_batch = total_agents

        # Create LSTM states for each batch
        for i in range(0, total_agents, agents_per_batch):
            batch_size = min(agents_per_batch, total_agents - i)
            self.lstm_h[i] = torch.zeros(num_lstm_layers, batch_size, hidden_size, device=self.device)
            self.lstm_c[i] = torch.zeros(num_lstm_layers, batch_size, hidden_size, device=self.device)

        # Minibatch configuration
        self.minibatch_size: int = min(minibatch_size, max_minibatch_size)
        self.accumulate_minibatches = max(1, minibatch_size // max_minibatch_size)

        minibatch_segments = self.minibatch_size / bptt_horizon
        self.minibatch_segments: int = int(minibatch_segments)
        if self.minibatch_segments != minibatch_segments:
            raise ValueError(f"minibatch_size {self.minibatch_size} must be divisible by bptt_horizon {bptt_horizon}")

        # Tracking for rollout completion
        self.full_rows = 0

        # Calculate num_minibatches for compatibility
        num_minibatches = self.segments / self.minibatch_segments
        self.num_minibatches: int = int(num_minibatches)
        if self.num_minibatches != num_minibatches:
            raise ValueError(
                f"Configuration error: segments ({self.segments}) must be divisible by "
                f"minibatch_segments ({self.minibatch_segments}).\n"
                f"segments = batch_size // bptt_horizon = {batch_size} // {bptt_horizon} = {self.segments}\n"
                f"minibatch_segments = minibatch_size // bptt_horizon = "
                f"{self.minibatch_size} // {bptt_horizon} = {self.minibatch_segments}\n"
                f"Please adjust trainer.minibatch_size in your configuration to ensure divisibility."
            )

        # Pre-allocate tensor to stores how many agents we have for use during environment reset
        self._range_tensor = torch.arange(total_agents, device=self.device, dtype=torch.int32)

    @property
    def ready_for_training(self) -> bool:
        """Check if buffer has enough data for training."""
        return self.full_rows >= self.segments

    def store(
        self,
        obs: Tensor,
        actions: Tensor,
        logprobs: Tensor,
        rewards: Tensor,
        dones: Tensor,
        truncations: Tensor,
        values: Tensor,
        env_id: slice,
        mask: Tensor,
        lstm_state: Optional[Dict[str, Tensor]] = None,
    ) -> int:
        """Store a batch of experience data."""
        assert isinstance(env_id, slice), (
            f"TypeError: env_id expected to be a slice for segmented storage. Got {type(env_id).__name__} instead."
        )

        num_steps = mask.sum().item()
        episode_length = self.ep_lengths[env_id.start].item()
        indices = self.ep_indices[env_id]

        # Store data in segmented tensors
        batch_slice = (indices, episode_length)
        self.obs[batch_slice] = obs
        self.actions[batch_slice] = actions
        self.logprobs[batch_slice] = logprobs
        self.rewards[batch_slice] = rewards
        self.dones[batch_slice] = dones.float()
        self.truncateds[batch_slice] = truncations.float()
        self.values[batch_slice] = values

        # Update episode tracking
        self.ep_lengths[env_id] += 1

        # Check if episodes are complete and reset if needed
        if episode_length + 1 >= self.bptt_horizon:
            self._reset_completed_episodes(env_id)

        # Update LSTM states if provided
        if lstm_state is not None and env_id.start in self.lstm_h:
            self.lstm_h[env_id.start] = lstm_state["lstm_h"]
            self.lstm_c[env_id.start] = lstm_state["lstm_c"]

        return int(num_steps)

    def _reset_completed_episodes(self, env_id: slice) -> None:
        """Reset episode tracking for completed episodes."""
        num_full = env_id.stop - env_id.start
        # Use pre-allocated range tensor and slice it
        self.ep_indices[env_id] = (self.free_idx + self._range_tensor[:num_full]) % self.segments
        self.ep_lengths[env_id] = 0
        self.free_idx = (self.free_idx + num_full) % self.segments
        self.full_rows += num_full

        if self.reset_lstm_state_between_episodes and env_id.start in self.lstm_h:
            self.lstm_h[env_id.start].zero_()
            self.lstm_c[env_id.start].zero_()

    def get_lstm_state(self, env_id_start: int) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Get LSTM state as tensors."""
        if env_id_start not in self.lstm_h:
            return None, None
        return self.lstm_h[env_id_start], self.lstm_c[env_id_start]

    def _get_lstm_states_for_segments(self, segment_indices: Tensor) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Get LSTM states for given segment indices."""
        if self.reset_lstm_state_between_episodes or not self.lstm_h:
            return None, None
        # For now, return None - proper LSTM state propagation across segments
        # would require tracking which agent owns each segment
        return None, None

    def set_lstm_state(self, env_id_start: int, lstm_h: Tensor, lstm_c: Tensor) -> None:
        """Set LSTM state."""
        if env_id_start in self.lstm_h:
            self.lstm_h[env_id_start] = lstm_h
            self.lstm_c[env_id_start] = lstm_c

    def reset_for_rollout(self) -> None:
        """Reset tracking variables for a new rollout."""
        self.full_rows = 0
        self.free_idx = self.total_agents % self.segments
        self.ep_indices = self._range_tensor % self.segments
        self.ep_lengths.zero_()

    def reset_importance_sampling_ratios(self) -> None:
        """Reset the importance sampling ratio to 1.0."""
        self.ratio.fill_(1.0)

    def sample_minibatch(
        self,
        advantages: Tensor,
        prio_alpha: float,
        prio_beta: float,
        minibatch_idx: int,
        total_minibatches: int,
    ) -> Dict[str, Tensor]:
        """Sample a prioritized minibatch for training."""
        # Prioritized sampling based on advantage magnitude
        adv_magnitude = advantages.abs().sum(dim=1)
        prio_weights = torch.nan_to_num(adv_magnitude**prio_alpha, 0, 0, 0)
        prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)

        # Sample segment indices
        idx = torch.multinomial(prio_probs, self.minibatch_segments)

        # Get minibatch data
        mb_obs = self.obs[idx]
        if self.cpu_offload:
            mb_obs = mb_obs.to(self.device, non_blocking=True)

        return {
            "obs": mb_obs,
            "actions": self.actions[idx],
            "logprobs": self.logprobs[idx],
            "values": self.values[idx],
            "rewards": self.rewards[idx],
            "dones": self.dones[idx],
            "advantages": advantages[idx],
            "returns": advantages[idx] + self.values[idx],
            "indices": idx,
            "prio_weights": (self.segments * prio_probs[idx, None]) ** -prio_beta,
            "ratio": self.ratio[idx],
        }

    def update_values(self, indices: Tensor, new_values: Tensor) -> None:
        """Update value estimates for given indices."""
        self.values[indices] = new_values.detach()

    def update_ratio(self, indices: Tensor, new_ratio: Tensor) -> None:
        """Update importance sampling ratios for given indices."""
        self.ratio[indices] = new_ratio.detach()

    def stats(self) -> Dict[str, float]:
        """Get mean values of all tracked buffers."""
        stats = {
            "rewards": self.rewards.mean().item(),
            "values": self.values.mean().item(),
            "logprobs": self.logprobs.mean().item(),
            "dones": self.dones.mean().item(),
            "truncateds": self.truncateds.mean().item(),
            "ratio": self.ratio.mean().item(),
        }

        # Add episode length stats for active episodes
        active_episodes = self.ep_lengths > 0
        if active_episodes.any():
            stats["ep_lengths"] = self.ep_lengths[active_episodes].float().mean().item()
        else:
            stats["ep_lengths"] = 0.0

        # Add action statistics based on action space type
        if self.actions.dtype in [torch.int32, torch.int64]:
            # For discrete actions, we can add distribution info
            stats["actions_mean"] = self.actions.float().mean().item()
            stats["actions_std"] = self.actions.float().std().item()
        else:
            # For continuous actions
            stats["actions_mean"] = self.actions.mean().item()
            stats["actions_std"] = self.actions.std().item()

        return stats
