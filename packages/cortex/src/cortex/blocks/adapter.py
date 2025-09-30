from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.blocks.base import BaseBlock
from cortex.blocks.registry import build_block, register_block
from cortex.cells.base import MemoryCell
from cortex.config import AdapterBlockConfig
from cortex.types import MaybeState, ResetMask, Tensor


@register_block(AdapterBlockConfig)
class AdapterBlock(BaseBlock):
    """Wraps another block and adds a trainable adapter path.

    Structure:
        input → wrapped_block → adapter_residual → output

    The adapter is a gated residual bottleneck MLP initialized to identity (gate=0).
    This allows inserting adapters into pretrained models without changing behavior at t=0.

    Formula:
        y = block_out + gate * Up(act(Down(LN(block_out))))

    Where gate starts at 0, making this an identity wrapper initially.
    """

    def __init__(self, config: AdapterBlockConfig, d_hidden: int, cell: MemoryCell | None = None) -> None:
        # Import here to avoid circular dependency
        from cortex.cells import build_cell

        # Build the cell for the base block
        base_cell_hidden_size = config.base_block.get_cell_hidden_size(d_hidden)
        base_cell_config = type(config.base_block.cell)(
            **{**config.base_block.cell.model_dump(), "hidden_size": base_cell_hidden_size}
        )
        base_cell = build_cell(base_cell_config)

        # Initialize with the base cell
        super().__init__(d_hidden=d_hidden, cell=base_cell)
        self.config = config

        # Build the wrapped block with its cell
        self.wrapped_block = build_block(config=config.base_block, d_hidden=d_hidden, cell=base_cell)

        # Adapter components
        self.ln = nn.LayerNorm(d_hidden)
        self.down = nn.Linear(d_hidden, config.bottleneck, bias=True)
        self.up = nn.Linear(config.bottleneck, d_hidden, bias=True)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Activation
        if config.activation == "gelu":
            self.act = nn.GELU()
        elif config.activation == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()

        # Gate starts at 0 => exact identity mapping at init
        if config.per_channel_gate:
            self.gate = nn.Parameter(torch.zeros(d_hidden))
        else:
            self.gate = nn.Parameter(torch.zeros(()))  # scalar

        # Identity-friendly init: zero weights on the up projection
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)

    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        """Initialize state by delegating to wrapped block."""
        wrapped_state = self.wrapped_block.init_state(batch=batch, device=device, dtype=dtype)
        return TensorDict({"wrapped": wrapped_state}, batch_size=[batch])

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        """Reset state by delegating to wrapped block."""
        if state is None:
            return None
        wrapped_state = state.get("wrapped", None)
        new_wrapped_state = self.wrapped_block.reset_state(wrapped_state, mask)
        if new_wrapped_state is None:
            return None
        batch_size = state.batch_size[0] if state.batch_size else (mask.shape[0] if mask is not None else 1)
        return TensorDict({"wrapped": new_wrapped_state}, batch_size=[batch_size])

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        # Extract wrapped block state
        wrapped_state = state.get("wrapped") if state is not None else None

        # Run wrapped block
        y, new_wrapped_state = self.wrapped_block(x, wrapped_state, resets=resets)

        # Apply adapter residual (identity at init)
        adapter_out = self.up(self.dropout(self.act(self.down(self.ln(y)))))

        # Apply gate (broadcasts scalar to match tensor shape)
        g = self.gate if self.gate.ndim > 0 else self.gate.expand_as(adapter_out)
        y_adapted = y + g * adapter_out

        # Get batch size for state
        batch_size = x.shape[0]
        return y_adapted, TensorDict({"wrapped": new_wrapped_state}, batch_size=[batch_size])


__all__ = ["AdapterBlock"]
