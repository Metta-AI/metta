from typing import Optional

import torch
from tensordict import TensorClass


class PolicyState(TensorClass):
    # lstm_h: Optional[torch.Tensor] = None # Removed LSTM hidden state
    # lstm_c: Optional[torch.Tensor] = None # Removed LSTM cell state

    memory_tokens: Optional[torch.Tensor] = None  # For Transformer's recurrent state
    # hidden: Optional[torch.Tensor] = None # Removed as per review, assuming memory_tokens is the sole recurrent state
