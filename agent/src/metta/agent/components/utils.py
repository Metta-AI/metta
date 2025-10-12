from __future__ import annotations

import torch


def zero_long(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Allocate a long tensor filled with zeros on the given device."""

    tensor = torch.empty(*shape, dtype=torch.long, device=device)
    tensor.fill_(0)
    return tensor

