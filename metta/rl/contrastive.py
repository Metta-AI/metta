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
        max_pairs_per_agent: int = 32,  # Limit pairs per agent for speed
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
            max_pairs_per_agent: Maximum number of pairs to generate per agent (default: 32)
            device: Device to run computations on
        """
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.geometric_p = geometric_p
        self.max_temporal_distance = max_temporal_distance
        self.logsumexp_reg_coef = logsumexp_reg_coef
        self.max_pairs_per_agent = max_pairs_per_agent
        self.device = device

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        ).to(device)

        # Cache for projected states to avoid recomputation
        self._projected_cache = None
        self._cache_seq_len = None
        self._cache_batch_size = None

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
        projected = self._get_cached_projection(hidden_states, batch_size, seq_len)

        # Generate positive pairs using geometric distribution
        positive_pairs = self._generate_positive_pairs(batch_size, seq_len)

        # Generate negative pairs using uniform distribution - same number as positive
        negative_pairs = self._generate_negative_pairs(batch_size, seq_len, len(positive_pairs))

        # Compute contrastive loss
        contrastive_loss = self._compute_contrastive_loss(
            projected, positive_pairs, negative_pairs
        )

        # Compute contrastive reward (inverse of loss for exploration)
        contrastive_reward = -contrastive_loss.detach()

        return contrastive_loss, contrastive_reward

    def _get_cached_projection(self, hidden_states: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """Get projected states with caching to avoid recomputation."""
        # Check if we can use cached projection
        if (self._projected_cache is not None and
            self._cache_batch_size == batch_size and
            self._cache_seq_len == seq_len and
            self._projected_cache.shape == hidden_states.shape):
            # Use cached projection if shapes match
            return self._projected_cache

        # Compute new projection and cache it
        projected = self.projection_head(hidden_states)  # (batch_size, seq_len, hidden_size)
        projected = F.normalize(projected, dim=-1)  # L2 normalize

        # Cache the projection
        self._projected_cache = projected
        self._cache_batch_size = batch_size
        self._cache_seq_len = seq_len

        return projected

    def _generate_positive_pairs(self, batch_size: int, seq_len: int) -> Tensor:
        """Generate positive pairs using geometric distribution with limited sampling."""
        if seq_len < 2:
            return torch.empty(0, 4, device=self.device, dtype=torch.long)

        # Limit the number of timesteps we sample from to reduce computation
        sample_timesteps = min(seq_len // 2, self.max_pairs_per_agent)
        if sample_timesteps < 1:
            sample_timesteps = 1

        geometric_dist = torch.distributions.Geometric(probs=self.geometric_p)
        temporal_distances = geometric_dist.sample((batch_size, sample_timesteps)).to(self.device)
        temporal_distances = temporal_distances.long()
        temporal_distances = torch.clamp(temporal_distances, 1, self.max_temporal_distance)

        positive_pairs = []
        for b in range(batch_size):
            step_size = max(1, seq_len // sample_timesteps)
            for i in range(sample_timesteps):
                t = i * step_size
                if t >= seq_len - 1:
                    break
                anchor_t = t
                positive_t = min(t + temporal_distances[b, i], seq_len - 1)
                if positive_t > anchor_t:
                    positive_pairs.append([b, anchor_t, b, positive_t])

        # Limit the number of positive pairs to max_pairs_per_agent * batch_size
        max_pairs = self.max_pairs_per_agent * batch_size
        if len(positive_pairs) > max_pairs:
            idx = torch.randperm(len(positive_pairs), device=self.device)[:max_pairs]
            positive_pairs = [positive_pairs[i] for i in idx.tolist()]

        if not positive_pairs:
            return torch.empty(0, 4, device=self.device, dtype=torch.long)

        positive_pairs_tensor = torch.tensor(positive_pairs, device=self.device, dtype=torch.long)
        return positive_pairs_tensor

    def _generate_negative_pairs(self, batch_size: int, seq_len: int, num_positive_pairs: int) -> Tensor:
        """Generate negative pairs using uniform distribution with limited sampling."""
        num_pairs = num_positive_pairs
        if num_pairs == 0:
            return torch.empty(0, 4, device=self.device, dtype=torch.long)
        anchor_batch = torch.randint(0, batch_size, (num_pairs,), device=self.device, dtype=torch.long)
        anchor_time = torch.randint(0, seq_len, (num_pairs,), device=self.device, dtype=torch.long)
        negative_batch = torch.randint(0, batch_size, (num_pairs,), device=self.device, dtype=torch.long)
        negative_time = torch.randint(0, seq_len, (num_pairs,), device=self.device, dtype=torch.long)
        negative_pairs = torch.stack([anchor_batch, anchor_time, negative_batch, negative_time], dim=1)
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
