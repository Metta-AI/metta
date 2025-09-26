from __future__ import annotations

import torch
from tensordict import TensorDict

Tensor = torch.Tensor
State = TensorDict
MaybeState = TensorDict | None
ResetMask = torch.Tensor


__all__ = ["MaybeState", "ResetMask", "State", "Tensor"]
