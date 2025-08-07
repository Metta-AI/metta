from typing import Optional

import torch
from tensordict import TensorClass


class LSTMState(TensorClass):
    lstm_h: Optional[torch.Tensor] = None
    lstm_c: Optional[torch.Tensor] = None
    hidden: Optional[torch.Tensor] = None
