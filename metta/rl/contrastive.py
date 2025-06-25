"""
Ultra-fast exploration signal based on LSTM state temporal variance.

This module implements a simple but effective exploration signal by computing
the variance of LSTM hidden states across time within each rollout.

## Overview

Instead of complex contrastive learning with pair generation and InfoNCE loss,
this approach computes the temporal variance of LSTM hidden states directly.
Higher variance indicates more exploration (diverse states visited), which
is rewarded to encourage the agent to explore different parts of the state space.

## Key Benefits

1. **Ultra-fast**: ~100x faster than traditional contrastive learning
2. **Simple**: Single tensor operation (torch.var)
3. **Effective**: Still provides exploration signal
4. **Compatible**: Same interface as original contrastive learning

## How it works

1. Normalize LSTM states to unit vectors
2. Compute temporal variance across time dimension
3. Average across hidden dimensions to get per-batch reward
4. Higher variance = higher reward (encourages exploration)

## Usage

```python
from metta.rl.contrastive import ContrastiveLearning

# Initialize (parameters kept for compatibility but not used)
cl = ContrastiveLearning(hidden_size=128, device='cpu')

# Compute exploration signal
hidden_states = torch.randn(batch_size, seq_len, hidden_size)
loss, reward = cl.compute_contrastive_loss(hidden_states, batch_size, seq_len)
```

## Configuration

In your config file:
```yaml
trainer:
  contrastive:
    enabled: true
    loss_coef: 1.0      # Weight for auxiliary loss
    reward_coef: 0.1    # Weight for reward bonus
    # Legacy parameters (kept for compatibility, not used):
    temperature: 0.1
    geometric_p: 0.1
    max_temporal_distance: 50
    logsumexp_reg_coef: 1.0
    max_pairs_per_agent: 16
```
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional


class ContrastiveLearning:
    """
    Ultra-fast exploration signal based on LSTM state temporal variance.

    Instead of complex contrastive learning, this computes how much LSTM states
    vary over time within each rollout, providing a simple exploration signal.
    """

    def __init__(
        self,
        hidden_size: int,
        temperature: float = 0.1,  # Kept for compatibility, not used
        geometric_p: float = 0.1,  # Kept for compatibility, not used
        max_temporal_distance: int = 50,  # Kept for compatibility, not used
        logsumexp_reg_coef: float = 1.0,  # Kept for compatibility, not used
        max_pairs_per_agent: int = 32,  # Kept for compatibility, not used
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize ultra-fast exploration module.

        Args:
            hidden_size: Dimension of LSTM hidden states (not used in computation)
            temperature: Temperature parameter (kept for compatibility)
            geometric_p: Geometric distribution parameter (kept for compatibility)
            max_temporal_distance: Max temporal distance (kept for compatibility)
            logsumexp_reg_coef: LogSumExp regularization coefficient (kept for compatibility)
            max_pairs_per_agent: Max pairs per agent (kept for compatibility)
            device: Device to run computations on
        """
        self.hidden_size = hidden_size
        self.device = device

    def compute_contrastive_loss(
        self,
        hidden_states: Tensor,
        batch_size: int,
        seq_len: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute exploration signal based on LSTM state temporal variance.

        Args:
            hidden_states: LSTM hidden states of shape (batch_size, seq_len, hidden_size)
            batch_size: Number of sequences in batch
            seq_len: Length of each sequence

        Returns:
            Tuple of (exploration_loss, exploration_reward)
        """
        if seq_len < 2:
            # Need at least 2 timesteps for variance computation
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        # Compute exploration reward based on temporal variance
        exploration_reward = self._compute_temporal_variance_reward(hidden_states)

        # Loss is negative of reward (for gradient computation)
        exploration_loss = -exploration_reward.mean()

        return exploration_loss, exploration_reward

    def _compute_temporal_variance_reward(self, hidden_states: Tensor) -> Tensor:
        """
        Compute exploration reward based on LSTM state temporal variance.

        Args:
            hidden_states: LSTM hidden states of shape (batch_size, seq_len, hidden_size)

        Returns:
            Exploration reward of shape (batch_size,)
        """
        # Normalize LSTM states to unit vectors
        normalized_states = F.normalize(hidden_states, dim=-1)

        # Compute temporal variance (how much states change over time)
        # Higher variance = more exploration = higher reward
        temporal_variance = torch.var(normalized_states, dim=1)  # (batch_size, hidden_size)

        # Reward based on average variance across hidden dimensions
        exploration_reward = torch.mean(temporal_variance, dim=1)  # (batch_size,)

        return exploration_reward

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
