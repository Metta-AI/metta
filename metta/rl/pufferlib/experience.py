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
        cpu_offload: bool = False,
        use_rnn: bool = False,
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
        agents_per_batch: Optional[int] = None,
    ):
        """Initialize experience buffer with segmented storage."""
        self.total_agents = total_agents
        self.batch_size = batch_size
        self.bptt_horizon = bptt_horizon
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.cpu_offload = cpu_offload
        self.use_rnn = use_rnn

        # Calculate segments
        self.segments = batch_size // bptt_horizon
        if total_agents > self.segments:
            raise ValueError(f"Total agents {total_agents} > segments {self.segments}")

        # Create segmented tensor storage
        obs_device = "cpu" if cpu_offload else self.device
        obs_dtype = torch.float32 if obs_space.dtype == np.float32 else torch.uint8

        self.observations = torch.zeros(
            self.segments,
            bptt_horizon,
            *obs_space.shape,
            dtype=obs_dtype,
            pin_memory=str(self.device).startswith("cuda") and cpu_offload,
            device=obs_device,
        )

        # Simplified action dtype handling
        self.actions = torch.zeros(
            self.segments,
            bptt_horizon,
            *atn_space.shape,
            device=self.device,
            dtype=torch.int32 if np.issubdtype(atn_space.dtype, np.integer) else torch.float32,
        )

        self.values = torch.zeros(self.segments, bptt_horizon, device=self.device)
        self.logprobs = torch.zeros(self.segments, bptt_horizon, device=self.device)
        self.rewards = torch.zeros(self.segments, bptt_horizon, device=self.device)
        self.terminals = torch.zeros(self.segments, bptt_horizon, device=self.device)
        self.truncations = torch.zeros(self.segments, bptt_horizon, device=self.device)
        self.ratio = torch.ones(self.segments, bptt_horizon, device=self.device)

        # Episode tracking
        self.ep_lengths = torch.zeros(total_agents, device=self.device, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, device=self.device, dtype=torch.int32) % self.segments
        self.free_idx = total_agents % self.segments

        # Simplified LSTM state management
        self.lstm_h: Dict[int, Tensor] = {}
        self.lstm_c: Dict[int, Tensor] = {}
        if use_rnn:
            # Use provided agents_per_batch or default to total_agents
            if agents_per_batch is None:
                agents_per_batch = total_agents

            # Create LSTM states for each batch
            for i in range(0, total_agents, agents_per_batch):
                batch_size = min(agents_per_batch, total_agents - i)
                self.lstm_h[i] = torch.zeros(num_lstm_layers, batch_size, hidden_size, device=self.device)
                self.lstm_c[i] = torch.zeros(num_lstm_layers, batch_size, hidden_size, device=self.device)

        # Minibatch configuration
        self.minibatch_size = min(minibatch_size, max_minibatch_size)
        self.accumulate_minibatches = max(1, minibatch_size // max_minibatch_size)
        self.minibatch_segments = self.minibatch_size // bptt_horizon

        # Tracking for rollout completion
        self.full_rows = 0

    def store(
        self,
        obs: Tensor,
        actions: Tensor,
        logprobs: Tensor,
        rewards: Tensor,
        terminals: Tensor,
        truncations: Tensor,
        values: Tensor,
        env_id: slice,
        mask: Tensor,
        lstm_state: Optional[Dict[str, Tensor]] = None,
    ) -> int:
        """Store a batch of experience."""
        num_steps = sum(mask)
        episode_length = self.ep_lengths[env_id.start].item()
        indices = self.ep_indices[env_id]

        # Store data - simplified indexing
        batch_slice = (indices, episode_length)
        self.observations[batch_slice] = obs if self.cpu_offload else obs
        self.actions[batch_slice] = actions
        self.logprobs[batch_slice] = logprobs
        self.rewards[batch_slice] = rewards
        self.terminals[batch_slice] = terminals.float()
        self.truncations[batch_slice] = truncations.float()
        self.values[batch_slice] = values

        # Update episode tracking
        self.ep_lengths[env_id] += 1

        # Check if episodes are complete
        if episode_length + 1 >= self.bptt_horizon:
            num_full = env_id.stop - env_id.start
            self.ep_indices[env_id] = (self.free_idx + torch.arange(num_full, device=self.device).int()) % self.segments
            self.ep_lengths[env_id] = 0
            self.free_idx = (self.free_idx + num_full) % self.segments
            self.full_rows += num_full

        # Update LSTM states if provided
        if lstm_state is not None and self.use_rnn and env_id.start in self.lstm_h:
            self.lstm_h[env_id.start] = lstm_state["lstm_h"]
            self.lstm_c[env_id.start] = lstm_state["lstm_c"]

        return int(num_steps)

    def get_lstm_state(self, env_id_start: int) -> Optional[Dict[str, Tensor]]:
        """Get LSTM state for a batch starting at env_id_start."""
        if not self.use_rnn or env_id_start not in self.lstm_h:
            return None
        return {"lstm_h": self.lstm_h[env_id_start], "lstm_c": self.lstm_c[env_id_start]}

    def reset_for_rollout(self) -> None:
        """Reset tracking variables for a new rollout."""
        self.full_rows = 0
        self.free_idx = self.total_agents % self.segments
        self.ep_indices = torch.arange(self.total_agents, device=self.device, dtype=torch.int32) % self.segments
        self.ep_lengths.zero_()

    def reset_ratio(self) -> None:
        """Reset the importance sampling ratio to 1.0."""
        self.ratio.fill_(1.0)

    def compute_advantages(
        self,
        gamma: float,
        gae_lambda: float,
        vtrace_rho_clip: float = 1.0,
        vtrace_c_clip: float = 1.0,
        average_reward: float = 0.0,
        use_average_reward: bool = False,
    ) -> Tensor:
        """Compute advantages using the pufferlib kernel."""
        advantages = torch.zeros_like(self.values)

        # Adjust rewards and gamma for average reward formulation
        rewards = self.rewards - average_reward if use_average_reward else self.rewards
        gamma = 1.0 if use_average_reward else gamma

        # Compute advantages using pufferlib kernel
        torch.ops.pufferlib.compute_puff_advantage(
            self.values,
            rewards,
            self.terminals,
            self.ratio,
            advantages,
            gamma,
            gae_lambda,
            vtrace_rho_clip,
            vtrace_c_clip,
        )

        return advantages

    def sample_minibatch(
        self,
        advantages: Tensor,
        prio_alpha: float,
        prio_beta: float,
        minibatch_idx: int,
        total_minibatches: int,
    ) -> Dict[str, Tensor]:
        """Sample a prioritized minibatch."""
        # Prioritized sampling based on advantage magnitude
        adv_magnitude = advantages.abs().sum(dim=1)
        prio_weights = torch.nan_to_num(adv_magnitude**prio_alpha, 0, 0, 0)
        prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)

        # Sample segment indices
        idx = torch.multinomial(prio_probs, self.minibatch_segments)

        # Compute importance sampling weights
        importance_weights = (self.segments * prio_probs[idx, None]) ** -prio_beta

        # Get minibatch data
        mb_obs = self.observations[idx]
        if self.cpu_offload:
            mb_obs = mb_obs.to(self.device, non_blocking=True)

        return {
            "obs": mb_obs,
            "actions": self.actions[idx],
            "logprobs": self.logprobs[idx],
            "values": self.values[idx],
            "rewards": self.rewards[idx],
            "terminals": self.terminals[idx],
            "advantages": advantages[idx],
            "returns": advantages[idx] + self.values[idx],
            "indices": idx,
            "prio_weights": importance_weights,
            "ratio": self.ratio[idx],
        }

    def update_values(self, indices: Tensor, new_values: Tensor) -> None:
        """Update value estimates for given indices."""
        self.values[indices] = new_values.detach()

    def update_ratio(self, indices: Tensor, new_ratio: Tensor) -> None:
        """Update importance sampling ratios for given indices."""
        self.ratio[indices] = new_ratio

    def get_mean_reward(self) -> float:
        """Get mean reward from the buffer."""
        return self.rewards.mean().item()

    @property
    def ready_for_training(self) -> bool:
        """Check if buffer has enough data for training."""
        return self.full_rows >= self.segments

    @property
    def num_minibatches(self) -> int:
        """Number of minibatches that can be created from the buffer."""
        return self.segments // self.minibatch_segments
