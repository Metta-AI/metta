"""Base abstract block interface for wrapping memory cells."""


import abc
import typing

import cortex.cells.base
import cortex.types
import tensordict
import torch
import torch.nn as nn


class BaseBlock(nn.Module, abc.ABC):
    """Abstract block wrapping a memory cell with optional projections."""

    def __init__(self, d_hidden: int, cell: cortex.cells.base.MemoryCell) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.cell = cell

    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> tensordict.TensorDict:
        cell_state = self.cell.init_state(batch=batch, device=device, dtype=dtype)
        cell_key = self.cell.__class__.__name__
        return tensordict.TensorDict({cell_key: cell_state}, batch_size=[batch])

    def reset_state(self, state: cortex.types.MaybeState, mask: cortex.types.ResetMask) -> cortex.types.MaybeState:
        if state is None:
            return None
        cell_key = self.cell.__class__.__name__
        cell_state = state.get(cell_key, None)
        new_cell_state = self.cell.reset_state(cell_state, mask)
        if new_cell_state is None:
            return None
        batch_size = (
            state.batch_size[0]
            if state.batch_size
            else (new_cell_state.batch_size[0] if new_cell_state.batch_size else mask.shape[0])
        )
        return tensordict.TensorDict({cell_key: new_cell_state}, batch_size=[batch_size])

    @abc.abstractmethod
    def forward(
        self,
        x: cortex.types.Tensor,
        state: cortex.types.MaybeState,
        *,
        resets: typing.Optional[cortex.types.ResetMask] = None,
    ) -> typing.Tuple[cortex.types.Tensor, cortex.types.MaybeState]: ...


__all__ = ["BaseBlock"]
