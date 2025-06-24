"""
Contrastive Unsupervised RL implementation for temporal contrastive learning.

This module implements contrastive learning on LSTM hidden states to encourage
better representations and exploration through temporal contrastive signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Tuple, Optional


class ContrastiveLearning:
    """
    Temporal contrastive learning module for LSTM hidden states.

    Implements contrastive learning where:
    - Positive pairs: Temporal distances drawn from geometric distribution
    - Negative pairs: Uniform sampling across rollout
    - Can be used both as auxiliary loss and reward signal
    """

    def __init__(
        self,
        hidden_size: int,
        temperature: float = 0.1,
        geometric_p: float = 0.1,
        max_temporal_distance: int = 50,
        logsumexp_reg_coef: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize contrastive learning module.

        Args:
            hidden_size: Dimension of LSTM hidden states
            temperature: Temperature for contrastive loss computation
            geometric_p: Success probability for geometric distribution (mean = 1/p)
            max_temporal_distance: Maximum temporal distance for positive pairs
            logsumexp_reg_coef: Coefficient for LogSumExp regularization (default: 1.0)
            device: Device to run computations on
        """
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.geometric_p = geometric_p
        self.max_temporal_distance = max_temporal_distance
        self.logsumexp_reg_coef = logsumexp_reg_coef
        self.device = device

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        ).to(device)

    def compute_contrastive_loss(
        self,
        hidden_states: Tensor,
        batch_size: int,
        seq_len: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute contrastive loss and reward signal.

        Args:
            hidden_states: LSTM hidden states of shape (batch_size, seq_len, hidden_size)
            batch_size: Number of sequences in batch
            seq_len: Length of each sequence

        Returns:
            Tuple of (contrastive_loss, contrastive_reward)
        """
        if seq_len < 2:
            # Need at least 2 timesteps for contrastive learning
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        # Project hidden states
        projected = self.projection_head(hidden_states)  # (batch_size, seq_len, hidden_size)
        projected = F.normalize(projected, dim=-1)  # L2 normalize

        # Generate positive pairs using geometric distribution
        positive_pairs = self._generate_positive_pairs(batch_size, seq_len)

        # Generate negative pairs using uniform distribution - same number as positive
        negative_pairs = self._generate_negative_pairs(batch_size, seq_len)

        # Compute contrastive loss
        contrastive_loss = self._compute_contrastive_loss(
            projected, positive_pairs, negative_pairs
        )

        # Compute contrastive reward (inverse of loss for exploration)
        contrastive_reward = -contrastive_loss.detach()

        return contrastive_loss, contrastive_reward

    def _generate_positive_pairs(self, batch_size: int, seq_len: int) -> Tensor:
        """Generate positive pairs using geometric distribution."""
        # Sample temporal distances from geometric distribution
        geometric_dist = torch.distributions.Geometric(probs=self.geometric_p)
        temporal_distances = geometric_dist.sample((batch_size, seq_len)).to(self.device)

        # Convert to long tensors for indexing
        temporal_distances = temporal_distances.long()

        # Clip to maximum distance
        temporal_distances = torch.clamp(temporal_distances, 1, self.max_temporal_distance)

        # Generate positive pair indices
        positive_pairs = []
        for b in range(batch_size):
            for t in range(seq_len):
                # Current timestep
                anchor_t = t
                # Positive timestep (within sequence bounds)
                positive_t = min(t + temporal_distances[b, t], seq_len - 1)
                if positive_t > anchor_t:  # Ensure positive pair is valid
                    positive_pairs.append([b, anchor_t, b, positive_t])

        if not positive_pairs:
            return torch.empty(0, 4, device=self.device, dtype=torch.long)

        positive_pairs_tensor = torch.tensor(positive_pairs, device=self.device, dtype=torch.long)
        return positive_pairs_tensor

    def _generate_negative_pairs(self, batch_size: int, seq_len: int) -> Tensor:
        """Generate negative pairs using uniform distribution."""
        # Generate exactly the same number of pairs as positive pairs
        num_pairs = batch_size * seq_len  # Target number of pairs

        # Random batch indices
        anchor_batch = torch.arange(batch_size, device=self.device).repeat_interleave(seq_len)
        anchor_time = torch.arange(seq_len, device=self.device).repeat(batch_size)

        # Random negative batch and time indices using uniform distribution
        uniform_batch = torch.distributions.Uniform(0, batch_size)
        uniform_time = torch.distributions.Uniform(0, seq_len)
        negative_batch = uniform_batch.sample((num_pairs,)).long().to(self.device)
        negative_time = uniform_time.sample((num_pairs,)).long().to(self.device)

        # Ensure all indices are long tensors for proper indexing
        anchor_batch = anchor_batch.long()
        anchor_time = anchor_time.long()
        negative_batch = negative_batch.long()
        negative_time = negative_time.long()

        # Stack into pairs
        negative_pairs = torch.stack([anchor_batch, anchor_time, negative_batch, negative_time], dim=1)

        # Ensure final tensor is long
        negative_pairs = negative_pairs.long()

        return negative_pairs

    def _compute_contrastive_loss(
        self,
        projected: Tensor,
        positive_pairs: Tensor,
        negative_pairs: Tensor
    ) -> Tensor:
        """Compute contrastive loss using InfoNCE with LogSumExp regularization."""
        if len(positive_pairs) == 0:
            return torch.tensor(0.0, device=self.device)

        # Extract anchor, positive, and negative embeddings
        anchor_indices = positive_pairs[:, :2]  # (batch, time)
        positive_indices = positive_pairs[:, 2:]  # (batch, time)
        negative_indices = negative_pairs[:, 2:]  # (batch, time)

        # Get embeddings
        anchors = projected[anchor_indices[:, 0], anchor_indices[:, 1]]  # (num_pairs, hidden_size)
        positives = projected[positive_indices[:, 0], positive_indices[:, 1]]  # (num_pairs, hidden_size)
        negatives = projected[negative_indices[:, 0], negative_indices[:, 1]]  # (num_pairs, hidden_size)

        # Compute similarities
        pos_sim = torch.sum(anchors * positives, dim=1) / self.temperature  # (num_pairs,)
        neg_sim = torch.sum(anchors * negatives, dim=1) / self.temperature  # (num_pairs,)

        # LogSumExp regularization for control tasks
        # This prevents the InfoNCE objective from becoming too sharp
        logsumexp_reg = torch.logsumexp(torch.stack([pos_sim, neg_sim], dim=1), dim=1)  # (num_pairs,)

        # InfoNCE loss with LogSumExp regularization
        # Original: -log(exp(pos_sim) / (exp(pos_sim) + exp(neg_sim)))
        # With regularization: -pos_sim + logsumexp_reg_coef * logsumexp_reg
        contrastive_loss = (-pos_sim + self.logsumexp_reg_coef * logsumexp_reg).mean()

        return contrastive_loss

    def get_lstm_hidden_states(self, policy_state: 'PolicyState') -> Optional[Tensor]:
        """
        Extract LSTM hidden states from policy state.

        Args:
            policy_state: Policy state containing LSTM states

        Returns:
            LSTM hidden states if available, None otherwise
        """
        if hasattr(policy_state, 'lstm_h') and policy_state.lstm_h is not None:
            return policy_state.lstm_h
        return None
