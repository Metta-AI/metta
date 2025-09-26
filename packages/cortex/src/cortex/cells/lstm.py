from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.cells.base import MemoryCell
from cortex.cells.registry import register_cell
from cortex.config import LSTMCellConfig
from cortex.types import MaybeState, ResetMask, Tensor


@register_cell(LSTMCellConfig)
class LSTMCell(MemoryCell):
    """Stateless wrapper over nn.LSTM with TensorDict state.

    State format (batch-first):
      - h: [B, num_layers, H_out]
      - c: [B, num_layers, H_cell] (H_cell == hidden_size)

    Input shapes: [B, T, H] (batch-first) or [B, H] for single step.
    Output shapes mirror inputs with last dim = H_out.
    """

    def __init__(self, cfg: LSTMCellConfig) -> None:
        super().__init__(hidden_size=cfg.hidden_size)
        self.cfg = cfg
        self.net = nn.LSTM(
            input_size=cfg.hidden_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            bias=cfg.bias,
            batch_first=True,  # Always batch-first
            dropout=cfg.dropout,
            proj_size=cfg.proj_size,
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

        # Get state tensors and transpose from [B, L, H] to [L, B, H] for nn.LSTM
        h = st.get("h").transpose(0, 1)  # [B, L, H] -> [L, B, H]
        c = st.get("c").transpose(0, 1)  # [B, L, H] -> [L, B, H]

        # Apply resets across time where provided
        if resets is not None:
            # Always expect batch-first resets: [B] or [B, T]
            if is_step:
                resets_bt = resets.reshape(-1, 1)  # [B] -> [B, 1]
            else:
                resets_bt = resets  # Already [B, T]

            # Iterate time steps to zero h/c for masked batches before step t
            if is_step:
                mask_b = resets_bt[:, 0].to(dtype=h.dtype).view(1, -1, 1)
                h = h * (1.0 - mask_b)
                c = c * (1.0 - mask_b)
                out, (hn, cn) = self.net(x_seq, (h, c))
            else:
                T = x_seq.shape[1]  # Always batch-first
                outputs = []
                h_t, c_t = h, c
                for t in range(T):
                    mask_b = resets_bt[:, t].to(dtype=h.dtype).view(1, -1, 1)
                    h_t = h_t * (1.0 - mask_b)
                    c_t = c_t * (1.0 - mask_b)
                    x_t = x_seq[:, t : t + 1]  # Always batch-first
                    out_t, (h_t, c_t) = self.net(x_t, (h_t, c_t))
                    outputs.append(out_t)
                out = torch.cat(outputs, dim=1)  # Always batch-first
                hn, cn = h_t, c_t
        else:
            out, (hn, cn) = self.net(x_seq, (h, c))

        # Convert output back to expected shape
        if is_step:
            y = out.squeeze(1)  # [B, 1, H] -> [B, H]
        else:
            y = out  # Already [B, T, H]

        # Transpose state back to batch-first: [L, B, H] -> [B, L, H]
        hn_bf = hn.transpose(0, 1)
        cn_bf = cn.transpose(0, 1)
        new_state = TensorDict({"h": hn_bf, "c": cn_bf}, batch_size=[batch_size])
        return y, new_state

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
