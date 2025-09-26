from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.cells.base import MemoryCell
from cortex.types import MaybeState, ResetMask, Tensor


class BaseBlock(nn.Module, ABC):
    """Abstract block wrapping a memory cell with optional projections."""

    def __init__(self, d_hidden: int, cell: MemoryCell) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.cell = cell

    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        cell_state = self.cell.init_state(batch=batch, device=device, dtype=dtype)
        return TensorDict({"cell": cell_state}, batch_size=[batch])

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        if state is None:
            return None
        cell_state = state.get("cell", None)
        new_cell_state = self.cell.reset_state(cell_state, mask)
        if new_cell_state is None:
            return None
        batch_size = (
            state.batch_size[0]
            if state.batch_size
            else (new_cell_state.batch_size[0] if new_cell_state.batch_size else mask.shape[0])
        )
        return TensorDict({"cell": new_cell_state}, batch_size=[batch_size])

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]: ...


__all__ = ["BaseBlock"]
