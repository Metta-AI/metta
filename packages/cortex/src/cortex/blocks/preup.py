from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from cortex.blocks.base import BaseBlock
from cortex.blocks.registry import register_block
from cortex.cells.base import MemoryCell
from cortex.config import PreUpBlockConfig
from cortex.types import MaybeState, ResetMask, Tensor


@register_block(PreUpBlockConfig)
class PreUpBlock(BaseBlock):
    """Project up before the cell, apply gated skip, then project down."""

    def __init__(self, config: PreUpBlockConfig, d_hidden: int, cell: MemoryCell) -> None:
        super().__init__(d_hidden=d_hidden, cell=cell)
        self.config = config
        self.d_inner = int(config.proj_factor * d_hidden)
        self.norm = nn.LayerNorm(d_hidden, elementwise_affine=True, bias=False)
        self.in_proj = nn.Linear(d_hidden, 2 * self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_hidden)
        self.act = nn.SiLU()
        self.learnable_skip = nn.Parameter(torch.ones(self.d_inner))
        assert cell.hidden_size == self.d_inner, "PreUpBlock requires cell.hidden_size == d_inner"

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        # Always expect batch-first input: [B, T, H] or [B, H]
        is_step = x.dim() == 2
        batch_size = x.shape[0]

        residual = x
        x_normed = self.norm(x)

        # Project input along feature dim and split into (a, z)
        if is_step:
            x_proj = self.in_proj(x_normed)
        else:
            # Always [B, T, H]
            B, T, H = x_normed.shape
            x_ = x_normed.reshape(B * T, H)
            x_proj = self.in_proj(x_).reshape(B, T, 2 * self.d_inner)

        a, z = torch.split(x_proj, split_size_or_sections=self.d_inner, dim=-1)
        a_act = self.act(a)

        from tensordict import TensorDict
        cell_state = state.get("cell", None) if state is not None else None
        y_inner, new_cell_state = self.cell(a, cell_state, resets=resets)

        # Gated skip and down-projection - always batch-first
        if is_step:
            y_skip = y_inner + (self.learnable_skip * a_act)
            y_gate = y_skip * self.act(z)
            y = self.out_proj(y_gate)
        else:
            # Always [B, T, H]
            B, T, H = y_inner.shape
            y_skip = y_inner + (self.learnable_skip * a_act)
            y_gate = y_skip * self.act(z)
            y_ = y_gate.reshape(B * T, H)
            y = self.out_proj(y_).reshape(B, T, self.d_hidden)

        y = residual + y
        return y, TensorDict({"cell": new_cell_state}, batch_size=[batch_size])


__all__ = ["PreUpBlock"]
