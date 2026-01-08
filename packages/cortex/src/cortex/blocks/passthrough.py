"""Passthrough block that applies cell directly without projections."""

from __future__ import annotations

from typing import Optional, Tuple

from tensordict import TensorDict

from cortex.blocks.base import BaseBlock
from cortex.blocks.registry import register_block
from cortex.cells.base import MemoryCell
from cortex.config import PassThroughBlockConfig
from cortex.types import MaybeState, ResetMask, Tensor


@register_block(PassThroughBlockConfig)
class PassThroughBlock(BaseBlock):
    """Applies the cell directly, preserving external hidden size."""

    def __init__(self, config: PassThroughBlockConfig, d_hidden: int, cell: MemoryCell) -> None:
        super().__init__(d_hidden=d_hidden, cell=cell)
        self.config = config
        assert cell.hidden_size == d_hidden, "PassThroughBlock requires cell.hidden_size == d_hidden"

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        # Extract cell state from block state
        cell_key = self.cell.__class__.__name__
        cell_state = state.get(cell_key, None) if state is not None else None
        batch_size = state.batch_size[0] if state is not None and state.batch_size else x.shape[0]

        y, new_cell_state = self.cell(x, cell_state, resets=resets)

        # Wrap cell state in block state
        return y, TensorDict({cell_key: new_cell_state}, batch_size=[batch_size])


__all__ = ["PassThroughBlock"]
