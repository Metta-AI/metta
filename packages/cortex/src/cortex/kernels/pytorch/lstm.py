"""PyTorch LSTM kernel helpers.

This module exposes the pure PyTorch reference implementation used by the
LSTM cell. Keeping the sequence logic here allows higher-level cells to
focus on state bookkeeping while preserving a clear ground-truth path for
future Triton ports.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _validate_inputs(
    x_seq: torch.Tensor,
    h0_bf: torch.Tensor,
    c0_bf: torch.Tensor,
    resets: Optional[torch.Tensor],
) -> None:
    if x_seq.dim() != 3:
        raise ValueError(f"Expected x_seq to be 3D [B, T, H], got shape {tuple(x_seq.shape)}")

    if h0_bf.dim() != 3:
        raise ValueError(f"Expected h0 to be 3D [B, L, Hp], got shape {tuple(h0_bf.shape)}")

    if c0_bf.dim() != 3:
        raise ValueError(f"Expected c0 to be 3D [B, L, H], got shape {tuple(c0_bf.shape)}")

    if resets is not None and resets.dim() != 2:
        raise ValueError(f"Expected resets to be 2D [B, T], got shape {tuple(resets.shape)}")


def lstm_sequence_pytorch(
    *,
    lstm: nn.LSTM,
    x_seq: torch.Tensor,
    h0_bf: torch.Tensor,
    c0_bf: torch.Tensor,
    resets: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run an ``nn.LSTM`` over a batch-first sequence with optional resets.

    Args:
        lstm: ``nn.LSTM`` module configured for batch-first inputs.
        x_seq: Input sequence of shape ``[B, T, H]`` (batch-first).
        h0_bf: Initial hidden state in batch-first layout ``[B, L, Hp]``.
        c0_bf: Initial cell state in batch-first layout ``[B, L, H]``.
        resets: Optional reset mask ``[B, T]``. When ``resets[b, t]`` evaluates
            truthy, the hidden and cell states for batch ``b`` are zeroed before
            processing timestep ``t``.

    Returns:
        A tuple ``(y_seq, hn_bf, cn_bf)`` where:
            ``y_seq`` is the output sequence ``[B, T, Hp]`` matching the module's
            projection size.
            ``hn_bf`` is the final hidden state in batch-first form ``[B, L, Hp]``.
            ``cn_bf`` is the final cell state in batch-first form ``[B, L, H]``.
    """

    _validate_inputs(x_seq, h0_bf, c0_bf, resets)

    B, T, _ = x_seq.shape

    # nn.LSTM expects [L, B, H] states. Clone to avoid mutating caller tensors.
    h_t = h0_bf.transpose(0, 1).contiguous()
    c_t = c0_bf.transpose(0, 1).contiguous()

    if resets is None:
        y_seq, (hn, cn) = lstm(x_seq, (h_t, c_t))
    else:
        if resets.shape[0] != B:
            raise ValueError(f"Reset mask batch dimension {resets.shape[0]} does not match input batch {B}")
        if resets.shape[1] != T:
            raise ValueError(f"Reset mask time dimension {resets.shape[1]} does not match input time {T}")

        outputs = []
        dtype = h_t.dtype
        for t in range(T):
            mask_b = resets[:, t].to(dtype=dtype).view(1, -1, 1)
            h_t = h_t * (1.0 - mask_b)
            c_t = c_t * (1.0 - mask_b)
            x_t = x_seq[:, t : t + 1]
            out_t, (h_t, c_t) = lstm(x_t, (h_t, c_t))
            outputs.append(out_t)
        y_seq = torch.cat(outputs, dim=1)
        hn, cn = h_t, c_t

    hn_bf = hn.transpose(0, 1)
    cn_bf = cn.transpose(0, 1)
    return y_seq, hn_bf, cn_bf


__all__ = ["lstm_sequence_pytorch"]
