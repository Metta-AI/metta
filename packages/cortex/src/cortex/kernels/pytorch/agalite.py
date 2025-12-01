"""PyTorch reference utilities for AGaLiTe cells."""

from __future__ import annotations

import torch


@torch.jit.script
def _jit_discounted_sum(start_state: torch.Tensor, x: torch.Tensor, discounts: torch.Tensor) -> torch.Tensor:
    """Time-major discounted sum without inplace ops."""
    T = x.shape[0]
    if T == 0:
        return x
    outputs = []  # type: ignore[var-annotated]
    current = discounts[0] * start_state + x[0]
    outputs.append(current)
    for t in range(1, T):
        current = discounts[t] * current + x[t]
        outputs.append(current)
    return torch.stack(outputs, dim=0)


def discounted_sum_pytorch(start_state: torch.Tensor, x: torch.Tensor, discounts: torch.Tensor) -> torch.Tensor:
    """Reference discounted sum with minimal overhead and broadcasting."""
    if start_state.dim() < x.dim() - 1:
        shape_diff = x.dim() - 1 - start_state.dim()
        for _ in range(shape_diff):
            start_state = start_state.unsqueeze(-1)
    return _jit_discounted_sum(start_state, x, discounts)


__all__ = ["discounted_sum_pytorch"]
