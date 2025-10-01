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
from cortex.types import MaybeState, ResetMask, Tensor


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
            else:
                # For sequences, reset at beginning
                # TODO: Handle per-timestep resets properly
                pass

        if is_step:
            # Step-by-step processing
            assert T == 1, f"Step mode expects T=1, got T={T}"

            # Update ring buffer: roll and append new input
            conv_state = torch.roll(conv_state, shifts=-1, dims=1)
            conv_state[:, -1:, :] = x

            # One-step causal output via padding-free conv1d.
            # Works for both depthwise (groups=F) and channel-mixing (groups=1).
            # Input: [B, F, KS]  -> Output: [B, F, 1]
            x_conv_in = conv_state.transpose(1, 2)  # [B, F, KS]
            y = torch.nn.functional.conv1d(
                x_conv_in,
                self.conv.weight,
                self.conv.bias if self.cfg.causal_conv_bias else None,
                stride=1,
                padding=0,
                dilation=1,
                groups=self.groups,
            ).transpose(1, 2)  # [B, 1, F]

            new_state = TensorDict({"conv": conv_state}, batch_size=[B])
            return y.squeeze(1), new_state

        else:
            # Full sequence processing
            y = x.transpose(1, 2)  # [B, T, F] -> [B, F, T]
            y = self.conv(y)  # [B, F, T+pad]

            if self.pad > 0:
                y = y[:, :, : -self.pad]  # Remove padding

            y = y.transpose(1, 2)  # [B, F, T] -> [B, T, F]

            # Update conv state with last kernel_size inputs
            KS = self.cfg.kernel_size
            if T >= KS:
                new_conv_state = x[:, -KS:, :].clone()
            else:
                # Pad with zeros if sequence is shorter than kernel
                pad_zeros = torch.zeros(B, KS - T, F, device=x.device, dtype=x.dtype)
                new_conv_state = torch.cat([pad_zeros, x], dim=1)

            new_state = TensorDict({"conv": new_conv_state}, batch_size=[B])
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
