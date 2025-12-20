"""PyTorch reference utilities for AGaLiTe cells."""

from __future__ import annotations

import torch
import torch.jit


def _jit_discounted_sum(start_state: torch.Tensor, x: torch.Tensor, discounts: torch.Tensor) -> torch.Tensor:
    """Time-major discounted sum with higher-precision accumulator for low-precision inputs."""
    T = x.shape[0]
    if T == 0:
        return x

    # Use float32 accumulator for fp16/bf16 to reduce numerical drift; otherwise keep original dtype.
    if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
        acc_dtype = torch.float32
    else:
        acc_dtype = x.dtype

    x_acc = x.to(acc_dtype)
    discounts_acc = discounts.to(acc_dtype)
    start_acc = start_state.to(acc_dtype)

    outputs = []  # type: ignore[var-annotated]
    current = discounts_acc[0] * start_acc + x_acc[0]
    outputs.append(current)
    for t in range(1, T):
        current = discounts_acc[t] * current + x_acc[t]
        outputs.append(current)
    out_acc = torch.stack(outputs, dim=0)
    return out_acc.to(x.dtype)


_jit_discounted_sum = torch.compile(_jit_discounted_sum)


def discounted_sum_pytorch(start_state: torch.Tensor, x: torch.Tensor, discounts: torch.Tensor) -> torch.Tensor:
    """Reference discounted sum with minimal overhead and broadcasting."""
    if start_state.dim() < x.dim() - 1:
        shape_diff = x.dim() - 1 - start_state.dim()
        for _ in range(shape_diff):
            start_state = start_state.unsqueeze(-1)
    return _jit_discounted_sum(start_state, x, discounts)


__all__ = ["discounted_sum_pytorch"]
