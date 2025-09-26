from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from cortex.blocks.base import BaseBlock
from cortex.blocks.registry import register_block
from cortex.cells.base import MemoryCell
from cortex.config import PostUpBlockConfig
from cortex.types import MaybeState, ResetMask, Tensor


@register_block(PostUpBlockConfig)
class PostUpBlock(BaseBlock):
    """Apply cell first, then two-layer projection out and back."""

    def __init__(self, config: PostUpBlockConfig, d_hidden: int, cell: MemoryCell) -> None:
        super().__init__(d_hidden=d_hidden, cell=cell)
        self.config = config
        self.d_inner = int(config.proj_factor * d_hidden)
        assert cell.hidden_size == d_hidden, "PostUpBlock requires cell.hidden_size == d_hidden"
        self.out1 = nn.Linear(d_hidden, self.d_inner)
        self.out2 = nn.Linear(self.d_inner, d_hidden)

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        from tensordict import TensorDict

        # Extract cell state from block state
        cell_state = state.get("cell", None) if state is not None else None
        batch_size = state.batch_size[0] if state is not None and state.batch_size else x.shape[0]

        y_in, new_cell_state = self.cell(x, cell_state, resets=resets)

        # Apply projections - always batch-first
        is_step = y_in.dim() == 2
        if is_step:
            y = self.out2(self.out1(y_in))
        else:
            # Always [B, T, H]
            B, T, H = y_in.shape
            y_ = self.out1(y_in.reshape(B * T, H))
            y = self.out2(y_).reshape(B, T, self.d_hidden)

        # Wrap cell state in block state
        return y, TensorDict({"cell": new_cell_state}, batch_size=[batch_size])


__all__ = ["PostUpBlock"]
