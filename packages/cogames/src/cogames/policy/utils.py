"""Shared helpers for policy implementations."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch

# LSTM state is represented as a tuple of (hidden, cell) tensors.
LSTMState = Tuple[torch.Tensor, torch.Tensor]
LSTMStateDict = Dict[str, torch.Tensor]


class LSTMStateAdapter:
    """Utility helpers to translate between tuple and dict LSTM states."""

    @staticmethod
    def _ensure_three_dims(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            return tensor
        if tensor.dim() == 2:
            return tensor.unsqueeze(0)
        if tensor.dim() == 1:
            return tensor.unsqueeze(0).unsqueeze(1)
        if tensor.dim() == 0:
            return tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return tensor

    @staticmethod
    def _align_layers(tensor: torch.Tensor, expected_layers: int) -> torch.Tensor:
        if tensor.shape[0] == expected_layers:
            return tensor.contiguous()
        if tensor.shape[1] == expected_layers:
            return tensor.transpose(0, 1).contiguous()
        return tensor.contiguous()

    @classmethod
    def normalize_tuple(
        cls,
        state: Optional[LSTMState],
        expected_layers: Optional[int] = None,
    ) -> Optional[LSTMState]:
        if state is None:
            return None
        hidden, cell = state
        hidden = cls._ensure_three_dims(hidden)
        cell = cls._ensure_three_dims(cell)
        if expected_layers is not None:
            hidden = cls._align_layers(hidden, expected_layers)
            cell = cls._align_layers(cell, expected_layers)
        return hidden.contiguous(), cell.contiguous()

    @classmethod
    def from_dict(
        cls,
        state: LSTMStateDict,
        expected_layers: Optional[int] = None,
    ) -> Optional[LSTMState]:
        if not state:
            return None
        hidden = state.get("lstm_h")
        cell = state.get("lstm_c")
        if hidden is None or cell is None:
            return None
        hidden = cls._ensure_three_dims(hidden)
        cell = cls._ensure_three_dims(cell)
        if expected_layers is None:
            hidden_layers = hidden.transpose(0, 1).contiguous()
            cell_layers = cell.transpose(0, 1).contiguous()
        else:
            hidden_layers = cls._align_layers(hidden, expected_layers)
            cell_layers = cls._align_layers(cell, expected_layers)
        return hidden_layers, cell_layers

    @classmethod
    def unpack(
        cls,
        state: Optional[Union[LSTMState, LSTMStateDict]],
        expected_layers: Optional[int] = None,
    ) -> Tuple[Optional[LSTMState], Optional[LSTMStateDict]]:
        if state is None:
            return None, None
        if isinstance(state, tuple):
            return cls.normalize_tuple(state, expected_layers), None
        if isinstance(state, dict):
            return cls.from_dict(state, expected_layers), state
        msg = f"Unsupported LSTM state container type: {type(state)!r}"
        raise TypeError(msg)

    @classmethod
    def update_dict(
        cls,
        target: LSTMStateDict,
        state: Optional[LSTMState],
        expected_layers: Optional[int] = None,
    ) -> None:
        target.clear()
        if state is None:
            return
        normalized = cls.normalize_tuple(state, expected_layers)
        if normalized is None:
            return
        hidden, cell = normalized
        target["lstm_h"] = hidden.transpose(0, 1).contiguous().detach()
        target["lstm_c"] = cell.transpose(0, 1).contiguous().detach()


__all__ = ["LSTMState", "LSTMStateDict", "LSTMStateAdapter"]
