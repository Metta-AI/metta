"""LSTM cell with TensorDict state and dual PyTorch/Triton backends."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.cells.base import MemoryCell
from cortex.cells.registry import register_cell
from cortex.config import LSTMCellConfig
from cortex.kernels.pytorch.lstm import lstm_sequence_pytorch
from cortex.kernels.triton.lstm import lstm_sequence_triton
from cortex.types import MaybeState, ResetMask, Tensor
from cortex.utils import select_backend


@register_cell(LSTMCellConfig)
class LSTMCell(MemoryCell):
    """Standard LSTM cell with TensorDict state and dual backends."""

    def __init__(self, cfg: LSTMCellConfig) -> None:
        super().__init__(hidden_size=cfg.hidden_size)
        if cfg.num_layers != 1:
            raise ValueError("LSTMCell currently supports num_layers == 1 for both backends")
        if cfg.proj_size not in (0, None):
            raise ValueError("LSTMCell Triton backend does not support proj_size > 0")
        self.cfg = cfg
        self.net = nn.LSTM(
            input_size=cfg.hidden_size,
            hidden_size=cfg.hidden_size,
            num_layers=1,
            bias=cfg.bias,
            batch_first=True,
            dropout=0.0,
            proj_size=0,
        )

    @property
    def out_hidden_size(self) -> int:
        return self.cfg.proj_size if self.cfg.proj_size > 0 else self.cfg.hidden_size

    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        B = batch
        L = self.cfg.num_layers
        H = self.cfg.hidden_size
        Hp = self.out_hidden_size
        # Batch-first state tensors
        h = torch.zeros(B, L, Hp, device=device, dtype=dtype)
        c = torch.zeros(B, L, H, device=device, dtype=dtype)
        return TensorDict({"h": h, "c": c}, batch_size=[B])

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        # Always expect batch-first input: [B, T, H] or [B, H]
        is_step = x.dim() == 2
        if is_step:
            # [B, H] -> [B, 1, H]
            x_seq = x.unsqueeze(1)
        else:
            # Already [B, T, H]
            x_seq = x

        # Prepare state tuple
        if state is None or ("h" not in state.keys() or "c" not in state.keys()):
            batch_size = x_seq.shape[0]
            st = self.init_state(batch=batch_size, device=x.device, dtype=x.dtype)
        else:
            st = state
            batch_size = st.batch_size[0] if st.batch_size else x_seq.shape[0]

        h0 = st.get("h")
        c0 = st.get("c")
        assert h0 is not None and c0 is not None, "LSTM state must contain 'h' and 'c' tensors"

        resets_bt: ResetMask | None
        if resets is None:
            resets_bt = None
        elif is_step:
            resets_bt = resets.reshape(-1, 1)
        else:
            resets_bt = resets

        backend_kwargs = {
            "lstm": self.net,
            "x_seq": x_seq.contiguous(),
            "h0_bf": h0,
            "c0_bf": c0,
            "resets": resets_bt,
        }

        allow_triton = (
            x_seq.is_cuda
            and x_seq.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and self.net.weight_ih_l0.shape[1] == self.cfg.hidden_size
            and self._hidden_size_power_of_two
        )

        backend_fn = select_backend(
            triton_fn=lstm_sequence_triton if allow_triton else None,
            pytorch_fn=lstm_sequence_pytorch,
            tensor=x_seq,
            allow_triton=allow_triton,
        )

        y_seq, hn_bf, cn_bf = backend_fn(**backend_kwargs)

        y = y_seq.squeeze(1) if is_step else y_seq
        new_state = TensorDict({"h": hn_bf, "c": cn_bf}, batch_size=[batch_size])
        return y, new_state

    @property
    def _hidden_size_power_of_two(self) -> bool:
        H = self.cfg.hidden_size
        return H > 0 and (H & (H - 1)) == 0

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        if state is None:
            return None
        h = state.get("h")
        c = state.get("c")
        if h is None or c is None:
            return state
        batch_size = state.batch_size[0] if state.batch_size else h.shape[0]
        # Mask shape: [B] -> [B, 1, 1] for broadcasting with [B, L, H]
        mask_b = mask.to(dtype=h.dtype).view(-1, 1, 1)
        h = h * (1.0 - mask_b)
        c = c * (1.0 - mask_b)
        return TensorDict({"h": h, "c": c}, batch_size=[batch_size])


__all__ = ["LSTMCell"]
