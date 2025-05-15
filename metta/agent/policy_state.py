from typing import Optional

import torch
from tensordict import TensorClass


class PolicyState(TensorClass):
    """
    A container for policy state information.

    Attributes:
        lstm_h: LSTM hidden state in layer-first format [num_layers, batch_size, hidden_size]
        lstm_c: LSTM cell state in layer-first format [num_layers, batch_size, hidden_size]
        hidden: Hidden state representation for non-LSTM components
    """

    # Store LSTM states in layer-first format [num_layers, batch_size, hidden_size]
    lstm_h: Optional[torch.Tensor] = None
    lstm_c: Optional[torch.Tensor] = None
    hidden: Optional[torch.Tensor] = None

    @classmethod
    def create(cls, batch_size: int, num_layers: int, hidden_size: int, device=None):
        """Create a new PolicyState with layer-first LSTM states."""
        state = cls()
        if batch_size > 0:
            # Create tensors in layer-first format [num_layers, batch_size, hidden_size]
            state.lstm_h = torch.zeros(num_layers, batch_size, hidden_size, device=device)
            state.lstm_c = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        return state
