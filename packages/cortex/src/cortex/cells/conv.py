"""Causal Convolutional layers for Cortex cells."""

import typing

import cortex.cells.base
import cortex.cells.registry
import cortex.config
import cortex.kernels.pytorch.conv1d
import cortex.types
import cortex.utils
import tensordict
import torch
import torch.nn as nn
import torch.nn.functional


@cortex.cells.registry.register_cell(cortex.config.CausalConv1dConfig)
class CausalConv1d(cortex.cells.base.MemoryCell):
    """Causal 1D convolution with depthwise or channel-mixing modes and stateful buffering."""

    def __init__(self, cfg: cortex.config.CausalConv1dConfig) -> None:
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

    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> tensordict.TensorDict:
        """Initialize convolution state buffer."""
        if self.cfg.kernel_size == 0:
            return tensordict.TensorDict({}, batch_size=[batch])

        conv_state = torch.zeros(batch, self.cfg.kernel_size, self.cfg.feature_dim, device=device, dtype=dtype)
        return tensordict.TensorDict({"conv": conv_state}, batch_size=[batch])

    def forward(
        self,
        x: cortex.types.Tensor,
        state: cortex.types.MaybeState,
        *,
        resets: typing.Optional[cortex.types.ResetMask] = None,
    ) -> typing.Tuple[cortex.types.Tensor, cortex.types.MaybeState]:
        """Apply causal convolution with optional resets."""
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

        # Triton is only available for channel-mixing mode
        triton_fn = "cortex.kernels.triton.conv1d:causal_conv1d_triton" if self.cfg.channel_mixing else None

        # Select backend at runtime
        backend_fn = cortex.utils.select_backend(
            triton_fn=triton_fn,
            pytorch_fn=cortex.kernels.pytorch.conv1d.causal_conv1d_pytorch,
            tensor=x,
            allow_triton=True,
        )

        if backend_fn == cortex.kernels.pytorch.conv1d.causal_conv1d_pytorch:
            # PyTorch backend (supports all modes)
            y, conv_state = backend_fn(
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
            # Triton backend (channel-mixing only)
            y, conv_state = backend_fn(
                conv_state=conv_state,
                x=x,
                weight=self.conv.weight,
                bias=self.conv.bias if self.cfg.causal_conv_bias else None,
                groups=self.groups,
                resets=resets if not is_step else None,
            )

        new_state = tensordict.TensorDict({"conv": conv_state}, batch_size=[B])
        if is_step:
            return y.squeeze(1), new_state
        return y, new_state

    def reset_state(self, state: cortex.types.MaybeState, mask: cortex.types.ResetMask) -> cortex.types.MaybeState:
        """Reset state for masked batch elements."""
        if state is None or "conv" not in state:
            return state

        mask_expanded = mask.to(dtype=state["conv"].dtype).view(-1, 1, 1)
        state["conv"] = state["conv"] * (1.0 - mask_expanded)
        return state


__all__ = ["CausalConv1d"]
