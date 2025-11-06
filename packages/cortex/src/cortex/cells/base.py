"""Abstract interface for memory cells with explicit TensorDict state management."""

import abc
import typing

import cortex.types
import tensordict
import torch
import torch.nn as nn


class MemoryCell(nn.Module, abc.ABC):
    """Abstract memory cell interface with explicit TensorDict state passing."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    @abc.abstractmethod
    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> tensordict.TensorDict:
        """Initialize zero state for given batch size."""

    @abc.abstractmethod
    def forward(
        self,
        x: cortex.types.Tensor,
        state: cortex.types.MaybeState,
        *,
        resets: typing.Optional[cortex.types.ResetMask] = None,
    ) -> typing.Tuple[cortex.types.Tensor, cortex.types.MaybeState]:
        """Process input with current state and optional resets."""

    @abc.abstractmethod
    def reset_state(self, state: cortex.types.MaybeState, mask: cortex.types.ResetMask) -> cortex.types.MaybeState:
        """Apply reset mask to zero selected batch elements."""


__all__ = ["MemoryCell"]
