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

from typing import Dict

import torch
from tensordict import TensorDict
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
        experience_spec: TensorDict,
        device: torch.device | str,
        cpu_offload: bool = False,
    ):
        """Initialize experience buffer with segmented storage."""
        # Store parameters
        self.total_agents = total_agents
        self.batch_size: int = batch_size
        self.bptt_horizon: int = bptt_horizon
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

        self.buffer: TensorDict = experience_spec.expand(self.segments, self.bptt_horizon).clone()
        if self.cpu_offload:
            # Offload obs to CPU after creation to save GPU memory
            if "obs" in self.buffer.keys():
                self.buffer["obs"] = self.buffer["obs"].to("cpu")

        # Episode tracking
        self.ep_lengths = torch.zeros(total_agents, device=self.device, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, device=self.device, dtype=torch.int32) % self.segments
        self.free_idx = total_agents % self.segments

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
    def full(self) -> bool:
        """Alias for ready_for_training for compatibility."""
        return self.ready_for_training

    @property
    def ready_for_training(self) -> bool:
        """Check if buffer has enough data for training."""
        return self.full_rows >= self.segments

    def store(self, data_td: TensorDict, env_id: slice) -> None:
        """Store a batch of experience."""
        assert isinstance(env_id, slice), (
            f"TypeError: env_id expected to be a slice for segmented storage. Got {type(env_id).__name__} instead."
        )

        episode_length = self.ep_lengths[env_id.start].item()
        indices = self.ep_indices[env_id]

        # Store data in segmented tensors
        self.buffer[indices, episode_length] = data_td

        # Update episode tracking
        self.ep_lengths[env_id] += 1

        # Check if episodes are complete and reset if needed
        if episode_length + 1 >= self.bptt_horizon:
            self._reset_completed_episodes(env_id)

    def _reset_completed_episodes(self, env_id: slice) -> None:
        """Reset episode tracking for completed episodes."""
        num_full = env_id.stop - env_id.start
        # Use pre-allocated range tensor and slice it
        self.ep_indices[env_id] = (self.free_idx + self._range_tensor[:num_full]) % self.segments
        self.ep_lengths[env_id] = 0
        self.free_idx = (self.free_idx + num_full) % self.segments
        self.full_rows += num_full

    def reset_for_rollout(self) -> None:
        """Reset tracking variables for a new rollout."""
        self.full_rows = 0
        self.free_idx = self.total_agents % self.segments
        self.ep_indices = self._range_tensor % self.segments
        self.ep_lengths.zero_()

    def reset_importance_sampling_ratios(self) -> None:
        """Reset the importance sampling ratio to 1.0."""
        if "ratio" in self.buffer.keys():
            self.buffer["ratio"].fill_(1.0)

    def sample_minibatch(
        self,
        advantages: Tensor,
        prio_alpha: float,
        prio_beta: float,  # av delete this
    ) -> TensorDict:
        """Sample a prioritized minibatch."""
        # Prioritized sampling based on advantage magnitude
        adv_magnitude = advantages.abs().sum(dim=1)
        prio_weights = torch.nan_to_num(adv_magnitude**prio_alpha, 0, 0, 0)
        prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)

        # Sample segment indices
        idx = torch.multinomial(prio_probs, self.minibatch_segments)

        minibatch_td = self.buffer[idx].clone()
        if self.cpu_offload:
            minibatch_td = minibatch_td.to(self.device, non_blocking=True)

        minibatch_td["advantages"] = advantages[idx]
        minibatch_td["returns"] = advantages[idx] + minibatch_td["values"]
        # minibatch_td["indices"] = idx.view(-1, 1)
        # minibatch_td["prio_weights"] = (self.segments * prio_probs[idx, None]) ** -prio_beta
        # minibatch_td["indices"] = idx.view(-1, 1).expand(-1, self.bptt_horizon)
        # prio_weights_val = (self.segments * prio_probs[idx, None]) ** -prio_beta
        # minibatch_td["prio_weights"] = prio_weights_val.expand(-1, self.bptt_horizon)
        return minibatch_td, idx

    def update(self, indices: Tensor, data_td: TensorDict) -> None:
        """Update buffer with new data for given indices."""
        self.buffer[indices].update(data_td.detach())

    def stats(self) -> Dict[str, float]:
        """Get mean values of all tracked buffers."""
        stats = {
            "rewards": self.buffer["rewards"].mean().item(),
            "values": self.buffer["values"].mean().item(),
            "logprobs": self.buffer["logprobs"].mean().item(),
            "dones": self.buffer["dones"].mean().item(),
            "truncateds": self.buffer["truncateds"].mean().item(),
        }
        if "ratio" in self.buffer.keys():
            stats["ratio"] = self.buffer["ratio"].mean().item()

        # Add episode length stats for active episodes
        active_episodes = self.ep_lengths > 0
        if active_episodes.any():
            stats["ep_lengths"] = self.ep_lengths[active_episodes].float().mean().item()
        else:
            stats["ep_lengths"] = 0.0

        # Add action statistics based on action space type
        if "actions" in self.buffer.keys():
            actions = self.buffer["actions"]
            if actions.dtype in [torch.int32, torch.int64]:
                # For discrete actions, we can add distribution info
                stats["actions_mean"] = actions.float().mean().item()
                stats["actions_std"] = actions.float().std().item()
            else:
                # For continuous actions
                stats["actions_mean"] = actions.mean().item()
                stats["actions_std"] = actions.std().item()

        return stats
