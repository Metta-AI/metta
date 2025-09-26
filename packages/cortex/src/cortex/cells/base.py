from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.types import MaybeState, ResetMask, Tensor


class MemoryCell(nn.Module, ABC):
    """Stateless memory cell interface used inside blocks.

    All state is passed as a TensorDict and returned alongside outputs.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    @abstractmethod
    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        """Return an initial state TensorDict for the given batch."""

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        """Apply the cell to input and state, returning output and new state."""

    @abstractmethod
    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        """Return a state with masked batch elements reset."""


__all__ = ["MemoryCell"]
