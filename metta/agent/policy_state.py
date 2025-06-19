from typing import NamedTuple, Optional

import torch


class PolicyState(NamedTuple):
    """A container for the policy's state. Use a tuple for JIT compatibility."""

    lstm_h: Optional[torch.Tensor]
    lstm_c: Optional[torch.Tensor]
