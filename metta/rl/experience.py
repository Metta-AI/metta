from typing import Dict, List, Optional

import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.modules.transformer_wrapper import SegmentMemoryRecord
from metta.common.util.collections import duplicates


class Experience:
    """Segmented tensor storage for RL experience with BPTT support."""

    def __init__(
        self,
        total_agents: int,
        batch_size: int,
        bptt_horizon: int,
        minibatch_size: int,
        max_minibatch_size: int,
        experience_spec: Composite,
        device: torch.device | str,
    ):
        """Initialize experience buffer with segmented storage."""
        self._check_for_duplicate_keys(experience_spec)

        # Store parameters
        self.total_agents = total_agents
        self.batch_size: int = batch_size
        self.bptt_horizon: int = bptt_horizon
        self.device = device if isinstance(device, torch.device) else torch.device(device)

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

        spec = experience_spec.expand(self.segments, self.bptt_horizon).to(self.device)
        self.buffer = spec.zero()

        # Episode tracking
        self.ep_lengths = torch.zeros(total_agents, device=self.device, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, device=self.device, dtype=torch.int32) % self.segments
        self.free_idx = total_agents % self.segments
        self.current_episode_ids = torch.zeros(total_agents, device=self.device, dtype=torch.long)
        self.segment_memory: list[Optional[Dict[str, Optional[List[torch.Tensor]]]]] = [None] * self.segments

        # Minibatch configuration
        self.minibatch_size: int = min(minibatch_size, max_minibatch_size)
        self.accumulate_minibatches = max(1, minibatch_size // max_minibatch_size)

        minibatch_segments = self.minibatch_size / bptt_horizon
        self.minibatch_segments: int = int(minibatch_segments)
        if self.minibatch_segments != minibatch_segments:
            raise ValueError(f"minibatch_size {self.minibatch_size} must be divisible by bptt_horizon {bptt_horizon}")

        # Tracking for rollout completion
        self.full_rows = 0

        # Calculate num_minibatches
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

        self._range_tensor = torch.arange(total_agents, device=self.device, dtype=torch.int32)

    def _check_for_duplicate_keys(self, experience_spec: Composite) -> None:
        """Check for duplicate keys in the experience spec."""
        all_keys = list(experience_spec.keys(include_nested=True, leaves_only=True))
        if duplicate_keys := duplicates(all_keys):
            raise ValueError(f"Duplicate keys found in experience_spec: {[str(d) for d in duplicate_keys]}")

    @property
    def ready_for_training(self) -> bool:
        """Check if buffer has enough data for training."""
        return self.full_rows >= self.segments

    def get_rollout_context(self, env_id: slice) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        segment_indices = self.ep_indices[env_id].clone()
        segment_pos = self.ep_lengths[env_id].clone()
        episode_ids = self.current_episode_ids[env_id].clone()
        return segment_indices, segment_pos, episode_ids

    def store(
        self,
        data_td: TensorDict,
        env_id: slice,
        segment_records: list[SegmentMemoryRecord] | None = None,
    ) -> None:
        """Store a batch of experience."""
        assert isinstance(env_id, slice), (
            f"TypeError: env_id expected to be a slice for segmented storage. Got {type(env_id).__name__} instead."
        )
        episode_lengths = self.ep_lengths[env_id.start].item()
        indices = self.ep_indices[env_id]

        for meta_key in ("_segment_indices", "_segment_pos", "_episode_ids", "_env_indices"):
            if meta_key in data_td.keys():
                del data_td[meta_key]

        self.buffer.update_at_(data_td.select(*self.buffer.keys(include_nested=True)), (indices, episode_lengths))

        if segment_records:
            for record in segment_records:
                self.segment_memory[record.segment_index] = record.memory

        self.ep_lengths[env_id] += 1

        if episode_lengths + 1 >= self.bptt_horizon:
            self._reset_completed_episodes(env_id)

        dones = data_td.get("dones", None)
        truncs = data_td.get("truncateds", None)
        if dones is not None or truncs is not None:
            dones_mask = (
                dones.reshape(-1).to(torch.bool)
                if dones is not None
                else torch.zeros(indices.shape[0], dtype=torch.bool, device=self.device)
            )
            trunc_mask = (
                truncs.reshape(-1).to(torch.bool)
                if truncs is not None
                else torch.zeros(indices.shape[0], dtype=torch.bool, device=self.device)
            )
            finished = torch.logical_or(dones_mask, trunc_mask)
            if finished.any():
                for offset, flag in enumerate(finished.tolist()):
                    if flag:
                        env_abs_idx = env_id.start + offset
                        self.current_episode_ids[env_abs_idx] += 1

    def _reset_completed_episodes(self, env_id) -> None:
        """Reset episode tracking for completed episodes."""
        num_full = env_id.stop - env_id.start
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

    def update(self, indices: Tensor, data_td: TensorDict) -> None:
        """Update buffer with new data for given indices."""
        self.buffer[indices].update(data_td)

    def reset_importance_sampling_ratios(self) -> None:
        """Reset the importance sampling ratio to 1.0."""
        if "ratio" in self.buffer.keys():
            self.buffer["ratio"].fill_(1.0)

    def stats(self) -> Dict[str, float]:
        """Get mean values of all tracked buffers."""
        stats = {
            "rewards": self.buffer["rewards"].mean().item(),
            "values": self.buffer["values"].mean().item(),
            "act_log_prob": self.buffer["act_log_prob"].mean().item(),
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

    def get_segment_memory(self, indices: torch.Tensor) -> list[Optional[Dict[str, Optional[List[torch.Tensor]]]]]:
        return [self.segment_memory[idx] for idx in indices.tolist()]

    def give_me_empty_md_td(self) -> TensorDict:
        return TensorDict(
            {},
            batch_size=(self.minibatch_segments, self.bptt_horizon),
            device=self.device,
        )
