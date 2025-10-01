"""Causal Convolutional layers for Cortex cells."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional
from tensordict import TensorDict

from cortex.cells.base import MemoryCell
from cortex.cells.registry import register_cell
from cortex.config import CausalConv1dConfig
from cortex.kernels.conv1d import TRITON_AVAILABLE, causal_conv1d_pytorch
from cortex.types import MaybeState, ResetMask, Tensor

if TRITON_AVAILABLE:
    from cortex.kernels.conv1d import causal_conv1d_triton


class CausalConv1d(MemoryCell):
    """Causal 1D Convolution cell with stateful processing support.

    This cell implements a causal depthwise convolution that can process
    sequences step-by-step while maintaining a convolution state buffer.

    The cell supports:
    - Full sequence processing with automatic padding
    - Step-by-step processing with state management
    - Depthwise (groups=feature_dim) or channel-mixing (groups=1) modes
    """

    def __init__(self, cfg: CausalConv1dConfig) -> None:
        super().__init__(hidden_size=cfg.feature_dim)
        self.cfg = cfg

        # Determine grouping for convolution
        self.groups = cfg.feature_dim if not cfg.channel_mixing else 1

        if cfg.kernel_size == 0:
            self.conv = None  # No-op for kernel_size=0
        else:
            self.pad = cfg.kernel_size - 1  # Padding for temporal causality
            self.conv = nn.Conv1d(
                in_channels=cfg.feature_dim,
                out_channels=cfg.feature_dim,
                kernel_size=cfg.kernel_size,
                padding=self.pad,
                groups=self.groups,
                bias=cfg.causal_conv_bias,
            )

        # Select backend: Triton if available and channel-mixing, else PyTorch
        if TRITON_AVAILABLE and cfg.channel_mixing:
            self.backend_fn = causal_conv1d_triton
        else:
            self.backend_fn = causal_conv1d_pytorch

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize convolution parameters."""
        if self.conv is not None:
            self.conv.reset_parameters()

    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        """Initialize convolution state buffer."""
        if self.cfg.kernel_size == 0:
            return TensorDict({}, batch_size=[batch])

        conv_state = torch.zeros(batch, self.cfg.kernel_size, self.cfg.feature_dim, device=device, dtype=dtype)
        return TensorDict({"conv": conv_state}, batch_size=[batch])

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        """Forward pass with optional state and reset handling.

        Args:
            x: Input tensor [B, T, F] or [B, F]
            state: Optional state TensorDict with conv buffer
            resets: Optional reset mask [B] or [B, T]

        Returns:
            Tuple of (output [B, T, F] or [B, F], new_state)
        """
        # Handle both [B, F] and [B, T, F] inputs
        is_step = x.dim() == 2
        if is_step:
            x = x.unsqueeze(1)  # [B, F] -> [B, 1, F]

        B, T, F = x.shape

        # No-op for kernel_size=0
        if self.cfg.kernel_size == 0:
            if is_step:
                return x.squeeze(1), state
            return x, state

        # Initialize or get state
        if state is None or "conv" not in state:
            st = self.init_state(batch=B, device=x.device, dtype=x.dtype)
        else:
            st = state

        conv_state = st.get("conv")  # [B, KS, F]

        # Apply resets if provided
        if resets is not None and conv_state is not None:
            if is_step:
                mask = resets.to(dtype=x.dtype).view(B, 1, 1)
                conv_state = conv_state * (1.0 - mask)
            elif resets.dim() == 1:
                # Batch-level resets for sequences: reset at beginning
                mask = resets.to(dtype=x.dtype).view(B, 1, 1)
                conv_state = conv_state * (1.0 - mask)

        # Use selected backend kernel
        assert self.conv is not None  # kernel_size > 0 guaranteed by early return

        # Use Triton only if tensors are on CUDA
        use_triton = self.backend_fn == causal_conv1d_triton and x.is_cuda

        if not use_triton:
            # PyTorch backend (supports all modes)
            y, conv_state = causal_conv1d_pytorch(
                conv_state=conv_state,
                x=x,
                weight=self.conv.weight,
                bias=self.conv.bias if self.cfg.causal_conv_bias else None,
                groups=self.groups,
                pad=self.pad,
                conv=self.conv,
                resets=resets if not is_step else None,
            )
        else:
            # Triton backend (channel-mixing only, requires per-timestep resets)
            y, conv_state = causal_conv1d_triton(
                conv_state=conv_state,
                x=x,
                weight=self.conv.weight,
                bias=self.conv.bias if self.cfg.causal_conv_bias else None,
                groups=self.groups,
                resets=resets if not is_step else None,
            )

        new_state = TensorDict({"conv": conv_state}, batch_size=[B])
        if is_step:
            return y.squeeze(1), new_state
        return y, new_state

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        """Reset state for masked batch elements."""
        if state is None or "conv" not in state:
            return state

        mask_expanded = mask.to(dtype=state["conv"].dtype).view(-1, 1, 1)
        state["conv"] = state["conv"] * (1.0 - mask_expanded)
        return state


# Register the cell if config is available
try:
    register_cell(CausalConv1dConfig)(CausalConv1d)
except (NameError, ImportError):
    # Config might not be defined yet
    pass


__all__ = ["CausalConv1d"]
