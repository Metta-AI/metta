"""Causal Conv1d kernel implementations.

This module contains kernel functions for causal 1D convolution operations,
including step-by-step processing, batched sequence processing, and handling
of per-timestep resets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional

from cortex.types import Tensor

# Lazy import of Triton implementation
try:
    from cortex.kernels.triton.conv1d import channelmix_causal_conv1d_with_resets_triton
except ImportError:
    channelmix_causal_conv1d_with_resets_triton = None  # type: ignore[assignment]


def causal_conv1d_triton(
    conv_state: Tensor,
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    groups: int,
    resets: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Triton-optimized causal conv1d for sequences with per-timestep resets.

    This function uses fused Triton kernels for efficient GPU computation.
    Only supports channel-mixing mode (groups=1) with per-timestep resets.

    Args:
        conv_state: Convolution state buffer [B, KS, F]
        x: Input sequence [B, T, F]
        weight: Convolution weight [F, F, KS] (channel-mixing only)
        bias: Optional convolution bias [F]
        groups: Must be 1 (channel-mixing mode)
        resets: Reset mask [B, T]

    Returns:
        Tuple of (output [B, T, F], updated conv_state [B, KS, F])

    Raises:
        RuntimeError: If Triton is not available or conditions aren't met
        ValueError: If groups != 1
    """
    if channelmix_causal_conv1d_with_resets_triton is None:
        raise RuntimeError("Triton is not available. Install triton package.")

    if groups != 1:
        raise ValueError(f"Triton kernel only supports groups=1, got groups={groups}")

    if resets is None or resets.dim() != 2:
        raise ValueError("Triton kernel requires per-timestep resets [B, T]")

    if not (x.is_cuda and x.is_contiguous() and weight.is_contiguous()):
        raise RuntimeError("Triton kernel requires CUDA contiguous tensors")

    B, T, F = x.shape
    KS = conv_state.shape[1]

    # Call Triton fused kernel
    y = channelmix_causal_conv1d_with_resets_triton(
        conv_state.contiguous(),
        x.contiguous(),
        weight.contiguous(),
        bias.contiguous() if bias is not None else None,
        resets.contiguous(),
    )

    # Update conv state with last kernel_size inputs
    if T >= KS:
        new_conv_state = x[:, -KS:, :].clone()
    else:
        pad_zeros = torch.zeros(B, KS - T, F, device=x.device, dtype=x.dtype)
        new_conv_state = torch.cat([pad_zeros, x], dim=1)

    return y, new_conv_state


def causal_conv1d_pytorch(
    conv_state: Tensor,
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    groups: int,
    pad: int,
    conv: nn.Conv1d | None = None,
    resets: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Unified causal conv1d kernel supporting step and sequence processing.

    This kernel handles three modes:
    1. Step mode: x has shape [B, 1, F] - single timestep processing
    2. Sequence with per-timestep resets: resets has shape [B, T] - scan-based
    3. Fast sequence: no per-timestep resets - optimized vectorized processing

    Args:
        conv_state: Convolution state buffer [B, KS, F]
        x: Input [B, 1, F] for step mode or [B, T, F] for sequence mode
        weight: Convolution weight [F_out, F_in/groups, KS]
        bias: Optional convolution bias [F_out]
        groups: Number of groups for grouped convolution
        pad: Padding amount (kernel_size - 1)
        conv: Optional Conv1d module for fast sequence processing
        resets: Optional per-timestep reset mask [B, T]

    Returns:
        Tuple of (output, updated conv_state)
    """
    is_step = x.shape[1] == 1
    B, T, F = x.shape
    if is_step:
        # Step mode: single timestep processing
        # Update ring buffer: roll and append new input
        conv_state = torch.roll(conv_state, shifts=-1, dims=1)
        conv_state[:, -1:, :] = x

        # One-step causal output via padding-free conv1d
        y = torch.nn.functional.conv1d(
            conv_state.transpose(1, 2),
            weight,
            bias,
            stride=1,
            padding=0,
            dilation=1,
            groups=groups,
        ).transpose(1, 2)  # [B, 1, F]

        return y, conv_state

    # Sequence mode with per-timestep resets
    if resets is not None and resets.dim() == 2 and resets.shape[:2] == (B, T):
        # Fallback: PyTorch scan through time
        y_steps = []
        for t in range(T):
            # Apply reset mask to conv_state before processing this timestep
            mask = resets[:, t].to(dtype=x.dtype).view(B, 1, 1)
            conv_state = conv_state * (1.0 - mask)
            # Update ring buffer: roll and append new input
            conv_state = torch.roll(conv_state, shifts=-1, dims=1)
            conv_state[:, -1:, :] = x[:, t : t + 1, :]

            # Compute output for this timestep
            y_t = torch.nn.functional.conv1d(
                conv_state.transpose(1, 2),
                weight,
                bias,
                stride=1,
                padding=0,
                dilation=1,
                groups=groups,
            ).transpose(1, 2)  # [B, 1, F]
            y_steps.append(y_t)

        y = torch.cat(y_steps, dim=1)  # [B, T, F]
        return y, conv_state

    # Fast path: no per-timestep resets, use vectorized processing
    if conv is None:
        raise ValueError("conv module required for fast sequence processing")

    KS = conv_state.shape[1]

    # Use PyTorch's optimized conv1d
    y = x.transpose(1, 2)  # [B, T, F] -> [B, F, T]
    y = conv(y)  # [B, F, T+pad]

    if pad > 0:
        y = y[:, :, :-pad]  # Remove padding

    y = y.transpose(1, 2)  # [B, F, T] -> [B, T, F]

    # Update conv state with last kernel_size inputs
    if T >= KS:
        new_conv_state = x[:, -KS:, :].clone()
    else:
        # Pad with zeros if sequence is shorter than kernel
        pad_zeros = torch.zeros(B, KS - T, F, device=x.device, dtype=x.dtype)
        new_conv_state = torch.cat([pad_zeros, x], dim=1)

    return y, new_conv_state


__all__ = ["causal_conv1d_pytorch", "causal_conv1d_triton"]
