"""Type aliases for tensors and stateful computation."""


import tensordict
import torch

Tensor = torch.Tensor
State = tensordict.TensorDict
MaybeState = tensordict.TensorDict | None
ResetMask = torch.Tensor


__all__ = ["MaybeState", "ResetMask", "State", "Tensor"]
