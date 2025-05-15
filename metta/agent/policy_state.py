from typing import Optional

import torch
from tensordict import TensorClass


class PolicyState(TensorClass):
    lstm_h: Optional[torch.Tensor] = None  # Will be shape [batch_size, num_layers, hidden_size]
    lstm_c: Optional[torch.Tensor] = None  # Will be shape [batch_size, num_layers, hidden_size]
    hidden: Optional[torch.Tensor] = None

    @classmethod
    def create(cls, batch_size: int, num_layers: int, hidden_size: int, device=None):
        """Create a new PolicyState with batch-first LSTM states."""
        state = cls()
        if batch_size > 0:
            state.lstm_h = torch.zeros(batch_size, num_layers, hidden_size, device=device)
            state.lstm_c = torch.zeros(batch_size, num_layers, hidden_size, device=device)
        return state
