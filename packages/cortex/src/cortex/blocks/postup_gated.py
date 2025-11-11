"""Post-processing block with GRU-style gating (GTrXL-inspired)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.blocks.base import BaseBlock
from cortex.blocks.registry import register_block
from cortex.cells.base import MemoryCell
from cortex.config import PostUpGatedBlockConfig
from cortex.types import MaybeState, ResetMask, Tensor


class _GRUGatingUnit(nn.Module):
    """GRU-style gating unit mixing residual and sublayer outputs (GTrXL-style)."""

    def __init__(self, hidden_size: int, bg: float = 2.0) -> None:
        super().__init__()
        H = hidden_size
        self.Wr = nn.Linear(H, H, bias=False)
        self.Ur = nn.Linear(H, H, bias=False)
        self.Wz = nn.Linear(H, H, bias=False)
        self.Uz = nn.Linear(H, H, bias=False)
        self.Wg = nn.Linear(H, H, bias=False)
        self.Ug = nn.Linear(H, H, bias=False)
        self.bg = nn.Parameter(torch.full((H,), bg))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(r * x))
        g = (1 - z) * x + z * h
        return g


@register_block(PostUpGatedBlockConfig)
class PostUpGatedBlock(BaseBlock):
    """Block that applies cell then FFN with GRU-style gated residuals."""

    def __init__(self, config: PostUpGatedBlockConfig, d_hidden: int, cell: MemoryCell) -> None:
        super().__init__(d_hidden=d_hidden, cell=cell)
        self.config = config
        self.d_inner = int(config.proj_factor * d_hidden)
        assert cell.hidden_size == d_hidden, "PostUpGatedBlock requires cell.hidden_size == d_hidden"

        self.norm1 = nn.LayerNorm(d_hidden, elementwise_affine=True, bias=False)
        self.norm2 = nn.LayerNorm(d_hidden, elementwise_affine=True, bias=False)

        # FFN
        self.ffn_in = nn.Linear(d_hidden, self.d_inner)
        self.act = nn.SiLU()
        self.ffn_out = nn.Linear(self.d_inner, d_hidden)

        # GRU-style gates (GTrXL-inspired)
        self.gate1 = _GRUGatingUnit(d_hidden, bg=float(config.gru_bias))
        self.gate2 = _GRUGatingUnit(d_hidden, bg=float(config.gru_bias))

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        cell_key = self.cell.__class__.__name__
        cell_state = state.get(cell_key, None) if state is not None else None
        batch_size = state.batch_size[0] if state is not None and state.batch_size else x.shape[0]

        # Sublayer 1: Cell with pre-norm, then gated with residual x
        x1 = self.norm1(x)
        y_cell, new_cell_state = self.cell(x1, cell_state, resets=resets)
        y = self.gate1(x, y_cell)

        # Sublayer 2: FFN with pre-norm on y, then gate with residual y
        y2 = self.norm2(y)
        is_step = y2.dim() == 2
        if is_step:
            ffn = self.ffn_out(self.act(self.ffn_in(y2)))
        else:
            B, T, H = y2.shape
            ffn = self.ffn_in(y2.reshape(B * T, H))
            ffn = self.act(ffn)
            ffn = self.ffn_out(ffn).reshape(B, T, self.d_hidden)

        out = self.gate2(y, ffn)
        return out, TensorDict({cell_key: new_cell_state}, batch_size=[batch_size])


__all__ = ["PostUpGatedBlock"]
