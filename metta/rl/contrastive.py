"""
Contrastive learning module for Metta using InfoNCE loss with LogSumExp regularization.

This module implements contrastive learning on LSTM hidden states to learn
representations that are predictive of future states.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class ContrastiveLearning:
    """
    Contrastive learning module using InfoNCE loss with LogSumExp regularization.

    Uses LSTM hidden states directly as representations for contrastive learning.
    Samples positive pairs using geometric distribution and negative pairs from
    within and across batches.
    """

    def __init__(
        self,
        hidden_size: int,
        gamma: float,
        temperature: float = 0.1,
        logsumexp_coef: float = 0.01,
        device: torch.device | None = None,
    ):
        """
        Initialize contrastive learning module.

        Args:
            hidden_size: Dimension of LSTM hidden states
            gamma: Discount factor for geometric distribution
            temperature: Temperature parameter for softmax
            logsumexp_coef: Coefficient for LogSumExp regularization
            device: Device to run computations on
        """
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.temperature = temperature
        self.logsumexp_coef = logsumexp_coef
        self.device = device if device is not None else torch.device("cpu")

        # Geometric distribution parameter for sampling future states
        self.geometric_p = 1.0 - gamma

    def sample_future_indices(self, current_indices: Tensor, bptt_horizon: int, batch_size: int) -> Tensor:
        """
        Sample future state indices using geometric distribution.

        Args:
            current_indices: Current timestep indices [batch_size]
            bptt_horizon: Maximum horizon for sampling
            batch_size: Size of the batch

        Returns:
            Future state indices [batch_size]
        """
        # Sample Δ ~ Geometric(1-γ)
        # Use inverse CDF method: Δ = floor(log(1-U) / log(γ)) where U ~ Uniform(0,1)
        u = torch.rand(batch_size, device=self.device)
        delta = torch.floor(torch.log(1 - u) / torch.log(self.gamma)).long()

        # Clamp to BPTT horizon
        delta = torch.clamp(delta, min=1, max=bptt_horizon - 1)

        # Compute future indices
        future_indices = current_indices + delta

        return future_indices

    def sample_negative_indices(
        self,
        current_indices: Tensor,
        num_rollout_negatives: int,
        num_cross_rollout_negatives: int,
        batch_size: int,
        bptt_horizon: int,
        num_segments: int,
    ) -> Tensor:
        """
        Sample negative indices from within and across rollouts.

        Args:
            current_indices: Current timestep indices [batch_size]
            num_rollout_negatives: Number of negatives from same rollout
            num_cross_rollout_negatives: Number of negatives from other rollouts
            batch_size: Size of the batch
            bptt_horizon: BPTT horizon
            num_segments: Number of segments in experience buffer

        Returns:
            Negative indices [batch_size, num_negatives]
        """
        # Get segment and timestep for current indices
        current_segments = current_indices // bptt_horizon
        current_timesteps = current_indices % bptt_horizon

        negative_indices = []

        # Sample from same rollout (different timesteps)
        if num_rollout_negatives > 0:
            for i in range(batch_size):
                segment = current_segments[i]
                timestep = current_timesteps[i]

                # Sample different timesteps from same segment
                available_timesteps = list(range(bptt_horizon))
                available_timesteps.remove(timestep.item())

                if len(available_timesteps) >= num_rollout_negatives:
                    rollout_negatives = torch.tensor(
                        np.random.choice(available_timesteps, num_rollout_negatives, replace=False), device=self.device
                    )
                else:
                    # If not enough timesteps, sample with replacement
                    rollout_negatives = torch.tensor(
                        np.random.choice(available_timesteps, num_rollout_negatives, replace=True), device=self.device
                    )

                rollout_indices = segment * bptt_horizon + rollout_negatives
                negative_indices.append(rollout_indices)

        # Sample from other rollouts
        if num_cross_rollout_negatives > 0:
            for i in range(batch_size):
                current_segment = current_segments[i]

                # Sample different segments
                available_segments = list(range(num_segments))
                available_segments.remove(current_segment.item())

                if len(available_segments) >= num_cross_rollout_negatives:
                    cross_segments = torch.tensor(
                        np.random.choice(available_segments, num_cross_rollout_negatives, replace=False),
                        device=self.device,
                    )
                else:
                    # If not enough segments, sample with replacement
                    cross_segments = torch.tensor(
                        np.random.choice(available_segments, num_cross_rollout_negatives, replace=True),
                        device=self.device,
                    )

                # Sample random timesteps from those segments
                cross_timesteps = torch.randint(0, bptt_horizon, (num_cross_rollout_negatives,), device=self.device)
                cross_indices = cross_segments * bptt_horizon + cross_timesteps
                negative_indices.append(cross_indices)

        # Combine all negatives
        if negative_indices:
            return torch.stack(negative_indices)  # [batch_size, num_negatives]
        else:
            return torch.empty(batch_size, 0, device=self.device, dtype=torch.long)

    def compute_infonce_loss(
        self,
        lstm_states: Tensor,  # [batch_size, hidden_size]
        positive_indices: Tensor,  # [batch_size]
        negative_indices: Tensor,  # [batch_size, num_negatives]
        all_lstm_states: Tensor,  # [total_states, hidden_size]
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute InfoNCE loss with LogSumExp regularization.

        Args:
            lstm_states: Current LSTM states [batch_size, hidden_size]
            positive_indices: Indices of positive future states [batch_size]
            negative_indices: Indices of negative states [batch_size, num_negatives]
            all_lstm_states: All LSTM states in the buffer [total_states, hidden_size]

        Returns:
            Loss tensor and metrics dictionary
        """
        batch_size = lstm_states.shape[0]
        num_negatives = negative_indices.shape[1] if negative_indices.numel() > 0 else 0

        # Get positive and negative states
        positive_states = all_lstm_states[positive_indices]  # [batch_size, hidden_size]

        if num_negatives > 0:
            # Flatten negative indices and get corresponding states
            flat_negative_indices = negative_indices.view(-1)  # [batch_size * num_negatives]
            negative_states = all_lstm_states[flat_negative_indices]  # [batch_size * num_negatives, hidden_size]
            negative_states = negative_states.view(
                batch_size, num_negatives, -1
            )  # [batch_size, num_negatives, hidden_size]
        else:
            negative_states = torch.empty(batch_size, 0, self.hidden_size, device=self.device)

        # Normalize states for cosine similarity
        lstm_states_norm = F.normalize(lstm_states, dim=-1)
        positive_states_norm = F.normalize(positive_states, dim=-1)

        if num_negatives > 0:
            negative_states_norm = F.normalize(negative_states, dim=-1)

        # Compute similarities
        positive_sim = torch.sum(lstm_states_norm * positive_states_norm, dim=-1) / self.temperature  # [batch_size]

        if num_negatives > 0:
            # Compute similarities with negative states
            # [batch_size, num_negatives]
            negative_sim = torch.sum(lstm_states_norm.unsqueeze(1) * negative_states_norm, dim=-1) / self.temperature

            # InfoNCE loss: -log(exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sim))))
            logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)  # [batch_size, 1 + num_negatives]

            # LogSumExp regularization term
            logsumexp_term = torch.logsumexp(negative_sim, dim=-1)  # [batch_size]

            # Full loss
            infonce_loss = F.cross_entropy(logits, torch.zeros(batch_size, device=self.device, dtype=torch.long))
            regularization_loss = self.logsumexp_coef * logsumexp_term.mean()

            total_loss = infonce_loss + regularization_loss

            # Compute metrics
            metrics = {
                "contrastive_infonce_loss": infonce_loss.item(),
                "contrastive_regularization_loss": regularization_loss.item(),
                "contrastive_total_loss": total_loss.item(),
                "contrastive_positive_sim": positive_sim.mean().item(),
                "contrastive_negative_sim": negative_sim.mean().item(),
                "contrastive_num_negatives": num_negatives,
            }
        else:
            # No negatives case - just use positive similarity as loss
            total_loss = -positive_sim.mean()
            metrics = {
                "contrastive_infonce_loss": total_loss.item(),
                "contrastive_regularization_loss": 0.0,
                "contrastive_total_loss": total_loss.item(),
                "contrastive_positive_sim": positive_sim.mean().item(),
                "contrastive_negative_sim": 0.0,
                "contrastive_num_negatives": 0,
            }

        return total_loss, metrics

    def compute_contrastive_reward(
        self,
        lstm_states: Tensor,
        positive_indices: Tensor,
        negative_indices: Tensor,
        all_lstm_states: Tensor,
    ) -> Tensor:
        """
        Compute contrastive reward based on similarity to positive future state.

        Args:
            lstm_states: Current LSTM states [batch_size, hidden_size]
            positive_indices: Indices of positive future states [batch_size]
            negative_indices: Indices of negative states [batch_size, num_negatives]
            all_lstm_states: All LSTM states in the buffer [total_states, hidden_size]

        Returns:
            Contrastive reward [batch_size]
        """
        # Get positive states
        positive_states = all_lstm_states[positive_indices]  # [batch_size, hidden_size]

        # Normalize states for cosine similarity
        lstm_states_norm = F.normalize(lstm_states, dim=-1)
        positive_states_norm = F.normalize(positive_states, dim=-1)

        # Compute positive similarity as reward
        positive_sim = torch.sum(lstm_states_norm * positive_states_norm, dim=-1)  # [batch_size]

        return positive_sim
