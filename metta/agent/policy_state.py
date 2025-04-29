import torch
from tensordict import TensorDict, tensorclass


@tensorclass
class PolicyState(TensorDict):
    lstm_h: torch.Tensor
    lstm_c: torch.Tensor
    hidden: torch.Tensor

    def __init__(self, lstm_h=None, lstm_c=None, hidden=None):
        super().__init__(
            {
                "lstm_h": lstm_h,
                "lstm_c": lstm_c,
                "hidden": hidden,
            }
        )

    @classmethod
    def create(cls):
        return cls(
            lstm_h=None,
            lstm_c=None,
            hidden=None,
        )
