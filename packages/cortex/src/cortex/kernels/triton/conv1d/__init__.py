"""Triton kernels for causal conv1d operations."""

from __future__ import annotations

import torch
from cortex.kernels.triton.conv1d.channel_mixing import (
    channelmix_causal_conv1d_with_resets_triton,
)
from cortex.types import Tensor


def causal_conv1d_triton(
    conv_state: Tensor,
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    groups: int,
    resets: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Triton-optimized causal conv1d for sequences with per-timestep resets."""
    if channelmix_causal_conv1d_with_resets_triton is None:
        raise RuntimeError("Triton is not available. Install triton package.")

    if groups != 1:
        raise ValueError(f"Triton kernel only supports groups=1, got groups={groups}")

    if resets is None or resets.dim() != 2:
        raise ValueError("Triton kernel requires per-timestep resets [B, T]")

    if not (x.is_cuda and x.is_contiguous() and weight.is_contiguous()):
        raise RuntimeError("Triton kernel requires CUDA contiguous tensors")

    B, T, F = x.shape
    kernel_size = conv_state.shape[1]

    y = channelmix_causal_conv1d_with_resets_triton(
        conv_state.contiguous(),
        x.contiguous(),
        weight.contiguous(),
        bias.contiguous() if bias is not None else None,
        resets.contiguous(),
    )

    if T >= kernel_size:
        new_conv_state = x[:, -kernel_size:, :].clone()
    else:
        pad_zeros = torch.zeros(B, kernel_size - T, F, device=x.device, dtype=x.dtype)
        new_conv_state = torch.cat([pad_zeros, x], dim=1)

    return y, new_conv_state


__all__ = ["channelmix_causal_conv1d_with_resets_triton", "causal_conv1d_triton"]
