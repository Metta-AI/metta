from typing import Optional

import torch


class PolicyState:
    """
    A container for policy state information.

    Attributes:
        lstm_h: LSTM hidden state in layer-first format [num_layers, batch_size, hidden_size]
        lstm_c: LSTM cell state in layer-first format [num_layers, batch_size, hidden_size]
        hidden: Hidden state representation for non-LSTM components
    """

    def __init__(
        self,
        lstm_h: Optional[torch.Tensor] = None,
        lstm_c: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
    ):
        self.lstm_h = lstm_h
        self.lstm_c = lstm_c
        self.hidden = hidden

    def __repr__(self):
        """String representation for debugging."""
        h_shape = None if self.lstm_h is None else self.lstm_h.shape
        c_shape = None if self.lstm_c is None else self.lstm_c.shape
        hidden_shape = None if self.hidden is None else self.hidden.shape

        return f"PolicyState(lstm_h shape={h_shape}, lstm_c shape={c_shape}, hidden shape={hidden_shape})"

    def to(self, device):
        """Move all tensors to the specified device."""
        if self.lstm_h is not None:
            self.lstm_h = self.lstm_h.to(device)
        if self.lstm_c is not None:
            self.lstm_c = self.lstm_c.to(device)
        if self.hidden is not None:
            self.hidden = self.hidden.to(device)
        return self

    def detach(self):
        """Detach all tensors from the computation graph."""
        if self.lstm_h is not None:
            self.lstm_h = self.lstm_h.detach()
        if self.lstm_c is not None:
            self.lstm_c = self.lstm_c.detach()
        if self.hidden is not None:
            self.hidden = self.hidden.detach()
        return self
