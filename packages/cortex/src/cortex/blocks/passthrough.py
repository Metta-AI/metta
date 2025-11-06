"""Passthrough block that applies cell directly without projections."""


import typing

import cortex.blocks.base
import cortex.blocks.registry
import cortex.cells.base
import cortex.config
import cortex.types
import tensordict


@cortex.blocks.registry.register_block(cortex.config.PassThroughBlockConfig)
class PassThroughBlock(cortex.blocks.base.BaseBlock):
    """Applies the cell directly, preserving external hidden size."""

    def __init__(
        self, config: cortex.config.PassThroughBlockConfig, d_hidden: int, cell: cortex.cells.base.MemoryCell
    ) -> None:
        super().__init__(d_hidden=d_hidden, cell=cell)
        self.config = config
        assert cell.hidden_size == d_hidden, "PassThroughBlock requires cell.hidden_size == d_hidden"

    def forward(
        self,
        x: cortex.types.Tensor,
        state: cortex.types.MaybeState,
        *,
        resets: typing.Optional[cortex.types.ResetMask] = None,
    ) -> typing.Tuple[cortex.types.Tensor, cortex.types.MaybeState]:
        # Extract cell state from block state
        cell_key = self.cell.__class__.__name__
        cell_state = state.get(cell_key, None) if state is not None else None
        batch_size = state.batch_size[0] if state is not None and state.batch_size else x.shape[0]

        y, new_cell_state = self.cell(x, cell_state, resets=resets)

        # Wrap cell state in block state
        return y, tensordict.TensorDict({cell_key: new_cell_state}, batch_size=[batch_size])


__all__ = ["PassThroughBlock"]
