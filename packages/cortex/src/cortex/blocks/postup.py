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
    """Apply LayerNorm, cell, then two-layer projection with residual connection."""

    def __init__(self, config: PostUpBlockConfig, d_hidden: int, cell: MemoryCell) -> None:
        super().__init__(d_hidden=d_hidden, cell=cell)
        self.config = config
        self.d_inner = int(config.proj_factor * d_hidden)
        assert cell.hidden_size == d_hidden, "PostUpBlock requires cell.hidden_size == d_hidden"
        self.norm = nn.LayerNorm(d_hidden, elementwise_affine=True, bias=False)
        self.ffn_norm = nn.LayerNorm(d_hidden, elementwise_affine=True, bias=False)
        self.out1 = nn.Linear(d_hidden, self.d_inner)
        self.act = nn.SiLU()
        self.out2 = nn.Linear(self.d_inner, d_hidden)

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        from tensordict import TensorDict

        cell_state = state.get("cell", None) if state is not None else None
        batch_size = state.batch_size[0] if state is not None and state.batch_size else x.shape[0]

        residual = x
        x_normed = self.norm(x)
        y_cell, new_cell_state = self.cell(x_normed, cell_state, resets=resets)
        y = residual + y_cell

        ffn_residual = y
        y_ffn_normed = self.ffn_norm(y)
        is_step = y_ffn_normed.dim() == 2
        if is_step:
            y_ffn = self.out1(y_ffn_normed)
            y_ffn = self.act(y_ffn)
            y_ffn = self.out2(y_ffn)
        else:
            B, T, H = y_ffn_normed.shape
            y_ = self.out1(y_ffn_normed.reshape(B * T, H))
            y_ = self.act(y_)
            y_ffn = self.out2(y_).reshape(B, T, self.d_hidden)

        y = ffn_residual + y_ffn
        return y, TensorDict({"cell": new_cell_state}, batch_size=[batch_size])


__all__ = ["PostUpBlock"]
