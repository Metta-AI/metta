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
        """Initialize experience buffer with segmented storage.

        Args:
            total_agents: Total number of agents in the environment
            batch_size: Total batch size for training
            bptt_horizon: Horizon for backpropagation through time
            minibatch_size: Size of minibatches for training
            max_minibatch_size: Maximum minibatch size (for gradient accumulation)
            obs_space: Observation space from the environment
            atn_space: Action space from the environment
            device: Device to store tensors on
            cpu_offload: Whether to offload observations to CPU
            use_rnn: Whether using RNN/LSTM policy
            hidden_size: Hidden size for LSTM states
            num_lstm_layers: Number of LSTM layers
            agents_per_batch: Number of agents per batch (from vecenv)
        """
        self.total_agents = total_agents
        self.batch_size = batch_size
        self.bptt_horizon = bptt_horizon
        self.device = device
        self.cpu_offload = cpu_offload
        self.use_rnn = use_rnn

        # Calculate segments
        self.segments = batch_size // bptt_horizon
        if total_agents > self.segments:
            raise ValueError(f"Total agents {total_agents} > segments {self.segments}")

        # Create segmented tensor storage
        obs_device = "cpu" if cpu_offload else device
        obs_dtype = torch.float32 if obs_space.dtype == np.float32 else torch.uint8
        atn_dtype = torch.int32 if atn_space.dtype in (np.int32, np.int64) else torch.float32

        self.observations = torch.zeros(
            self.segments,
            bptt_horizon,
            *obs_space.shape,
            dtype=obs_dtype,
            pin_memory=device == "cuda" and cpu_offload,
            device=obs_device,
        )
        self.actions = torch.zeros(
            self.segments,
            bptt_horizon,
            *atn_space.shape,
            device=device,
            dtype=atn_dtype,
        )

        self.values = torch.zeros(self.segments, bptt_horizon, device=device)
        self.logprobs = torch.zeros(self.segments, bptt_horizon, device=device)
        self.rewards = torch.zeros(self.segments, bptt_horizon, device=device)
        self.terminals = torch.zeros(self.segments, bptt_horizon, device=device)
        self.truncations = torch.zeros(self.segments, bptt_horizon, device=device)
        self.ratio = torch.ones(self.segments, bptt_horizon, device=device)
        self.importance = torch.ones(self.segments, bptt_horizon, device=device)

        # Episode tracking
        self.ep_lengths = torch.zeros(total_agents, device=device, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, device=device, dtype=torch.int32) % self.segments
        self.free_idx = total_agents % self.segments

        # LSTM states
        if use_rnn:
            # Use provided agents_per_batch or calculate it
            if agents_per_batch is None:
                agents_per_batch = max(1, total_agents // max(1, (total_agents // minibatch_size)))
            num_batches = total_agents // agents_per_batch
            self.lstm_h: Dict[int, Tensor] = {
                i * agents_per_batch: torch.zeros(num_lstm_layers, agents_per_batch, hidden_size, device=device)
                for i in range(num_batches)
            }
            self.lstm_c: Dict[int, Tensor] = {
                i * agents_per_batch: torch.zeros(num_lstm_layers, agents_per_batch, hidden_size, device=device)
                for i in range(num_batches)
            }
        else:
            self.lstm_h = {}
            self.lstm_c = {}

        # Minibatch configuration
        self.minibatch_size = min(minibatch_size, max_minibatch_size)
        if minibatch_size > max_minibatch_size and minibatch_size % max_minibatch_size != 0:
            raise ValueError(
                f"minibatch_size {minibatch_size} > max_minibatch_size {max_minibatch_size} must divide evenly"
            )

        if batch_size < minibatch_size:
            raise ValueError(f"batch_size {batch_size} must be >= minibatch_size {minibatch_size}")

        self.accumulate_minibatches = max(1, minibatch_size // max_minibatch_size)
        self.minibatch_segments = self.minibatch_size // bptt_horizon

        if self.minibatch_segments * bptt_horizon != self.minibatch_size:
            raise ValueError(f"minibatch_size {self.minibatch_size} must be divisible by bptt_horizon {bptt_horizon}")

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
        """Store a batch of experience.

        Args:
            obs: Observations tensor
            actions: Actions tensor
            logprobs: Log probabilities of actions
            rewards: Rewards tensor
            terminals: Terminal flags tensor
            truncations: Truncation flags tensor
            values: Value estimates tensor
            env_id: Environment ID slice
            mask: Mask for valid data
            lstm_state: Optional LSTM states

        Returns:
            Number of steps stored
        """
        num_steps = sum(mask)
        episode_length = self.ep_lengths[env_id.start].item()
        indices = self.ep_indices[env_id]

        # Store data
        obs_to_store = obs if self.cpu_offload else obs
        self.observations[indices, episode_length] = obs_to_store
        self.logprobs[indices, episode_length] = logprobs
        self.rewards[indices, episode_length] = rewards
        self.terminals[indices, episode_length] = terminals.float()
        self.truncations[indices, episode_length] = truncations.float()
        self.values[indices, episode_length] = values

        # Handle different action dtypes
        if self.actions.dtype == torch.int32:
            self.actions[indices, episode_length] = actions.int()
        elif self.actions.dtype == torch.int64:
            self.actions[indices, episode_length] = actions.long()
        else:
            self.actions[indices, episode_length] = actions

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
        if lstm_state is not None and self.use_rnn:
            batch_key = env_id.start
            if "lstm_h" in lstm_state and lstm_state["lstm_h"] is not None:
                self.lstm_h[batch_key] = lstm_state["lstm_h"]
                self.lstm_c[batch_key] = lstm_state["lstm_c"]

        return int(num_steps)

    def get_lstm_state(self, env_id_start: int) -> Optional[Dict[str, Tensor]]:
        """Get LSTM state for a batch starting at env_id_start.

        Args:
            env_id_start: Starting environment ID

        Returns:
            Dictionary with lstm_h and lstm_c if available, None otherwise
        """
        if not self.use_rnn:
            return None

        batch_key = env_id_start
        if batch_key in self.lstm_h:
            return {"lstm_h": self.lstm_h[batch_key], "lstm_c": self.lstm_c[batch_key]}
        return None

    def reset_for_rollout(self) -> None:
        """Reset tracking variables for a new rollout."""
        self.full_rows = 0
        self.free_idx = self.total_agents % self.segments
        self.ep_indices = torch.arange(self.total_agents, device=self.device, dtype=torch.int32) % self.segments
        self.ep_lengths.zero_()

    def compute_advantages(
        self,
        gamma: float,
        gae_lambda: float,
        vtrace_rho_clip: float = 1.0,
        vtrace_c_clip: float = 1.0,
        average_reward: float = 0.0,
        use_average_reward: bool = False,
    ) -> Tensor:
        """Compute advantages using the pufferlib kernel.

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            vtrace_rho_clip: V-trace rho clipping parameter
            vtrace_c_clip: V-trace c clipping parameter
            average_reward: Average reward for average reward formulation
            use_average_reward: Whether to use average reward

        Returns:
            Advantages tensor
        """
        shape = self.values.shape
        advantages = torch.zeros(shape, device=self.device)

        # Adjust rewards for average reward if needed
        if use_average_reward:
            rewards_adjusted = self.rewards - average_reward
            effective_gamma = 1.0
        else:
            rewards_adjusted = self.rewards
            effective_gamma = gamma

        # Compute advantages using pufferlib kernel
        torch.ops.pufferlib.compute_puff_advantage(
            self.values,
            rewards_adjusted,
            self.terminals,
            self.ratio,
            advantages,
            effective_gamma,
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
        """Sample a prioritized minibatch.

        Args:
            advantages: Advantages tensor
            prio_alpha: Prioritization alpha parameter
            prio_beta: Prioritization beta parameter
            minibatch_idx: Current minibatch index
            total_minibatches: Total number of minibatches

        Returns:
            Dictionary containing minibatch data
        """
        # Prioritized sampling
        adv = advantages.abs().sum(dim=1)
        prio_weights = torch.nan_to_num(adv**prio_alpha, 0, 0, 0)
        prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
        idx = torch.multinomial(prio_probs, self.minibatch_segments)
        mb_prio = (self.segments * prio_probs[idx, None]) ** -prio_beta

        # Get minibatch data
        mb_obs = self.observations[idx]
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
            "prio_weights": mb_prio,
            "ratio": self.ratio[idx],
        }

    def update_values(self, indices: Tensor, new_values: Tensor) -> None:
        """Update value estimates for given indices.

        Args:
            indices: Indices to update
            new_values: New value estimates
        """
        self.values[indices] = new_values.detach().float()

    def update_ratio(self, indices: Tensor, new_ratio: Tensor) -> None:
        """Update importance sampling ratios for given indices.

        Args:
            indices: Indices to update
            new_ratio: New importance sampling ratios
        """
        self.ratio[indices] = new_ratio

    def get_mean_reward(self) -> float:
        """Get mean reward from the buffer."""
        return float(self.rewards.mean().item())

    @property
    def ready_for_training(self) -> bool:
        """Check if buffer has enough data for training."""
        return self.full_rows >= self.segments
