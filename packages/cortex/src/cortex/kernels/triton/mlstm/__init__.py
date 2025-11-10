#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

"""Minimal Triton kernels for mLSTM from mlstm_kernels package."""

from __future__ import annotations

import math
from typing import Optional

import torch as pt

__version__ = "2.0.1"


def mlstm_chunkwise_triton(
    queries: pt.Tensor,
    keys: pt.Tensor,
    values: pt.Tensor,
    igate_preact: pt.Tensor,
    fgate_preact: pt.Tensor,
    initial_C: Optional[pt.Tensor] = None,
    initial_n: Optional[pt.Tensor] = None,
    initial_m: Optional[pt.Tensor] = None,
    reset_mask: Optional[pt.Tensor] = None,
    chunk_size: int = 64,
    return_last_state: bool = False,
    eps: float = 1e-6,
    **kwargs,
) -> tuple[pt.Tensor, tuple[pt.Tensor, pt.Tensor, pt.Tensor]]:
    """Triton-accelerated chunkwise mLSTM implementation."""
    try:
        from cortex.kernels.triton.mlstm.torch import mlstm_chunkwise__xl_chunk
    except ImportError:
        from cortex.kernels.pytorch.mlstm import mlstm_chunkwise_simple

        return mlstm_chunkwise_simple(
            queries=queries,
            keys=keys,
            values=values,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            initial_C=initial_C,
            initial_n=initial_n,
            initial_m=initial_m,
            reset_mask=reset_mask,
            chunk_size=chunk_size,
            return_last_state=return_last_state,
            eps=eps,
            **kwargs,
        )

    batch, num_heads, seq_len, hidden = queries.shape
    dtype, device = queries.dtype, queries.device

    c_initial = (
        initial_C if initial_C is not None else pt.zeros((batch, num_heads, hidden, hidden), dtype=dtype, device=device)
    )
    n_initial = initial_n if initial_n is not None else pt.zeros((batch, num_heads, hidden), dtype=dtype, device=device)
    if n_initial.dim() == 4:
        n_initial = n_initial.squeeze(-1)
    m_initial = initial_m if initial_m is not None else pt.zeros((batch, num_heads, 1), dtype=dtype, device=device)
    if m_initial.dim() == 4:
        m_initial = m_initial.squeeze(-1)

    original_seq_len = seq_len
    pad = (16 - (original_seq_len % 16)) % 16
    if pad > 0:
        zeros = pt.zeros(batch, num_heads, pad, hidden, dtype=dtype, device=device)
        q_pad = pt.cat([queries, zeros], dim=2)
        k_pad = pt.cat([keys, zeros], dim=2)
        v_pad = pt.cat([values, zeros], dim=2)

        pos_val = pt.tensor(10.0 if dtype in (pt.float16, pt.bfloat16) else 20.0, dtype=dtype, device=device)
        neg_val = -pos_val
        i_tail = neg_val.expand(batch, num_heads, pad)
        f_tail = pos_val.expand(batch, num_heads, pad)
        i_pad = pt.cat([igate_preact, i_tail], dim=2)
        f_pad = pt.cat([fgate_preact, f_tail], dim=2)

        reset_pad = None
        if reset_mask is not None:
            rm_tail = pt.zeros(batch, pad, dtype=reset_mask.dtype, device=device)
            reset_pad = pt.cat([reset_mask, rm_tail], dim=1)

        result = mlstm_chunkwise__xl_chunk(
            q=q_pad,
            k=k_pad,
            v=v_pad,
            i=i_pad,
            f=f_pad,
            c_initial=c_initial,
            n_initial=n_initial,
            m_initial=m_initial,
            return_last_states=return_last_state,
            eps=eps,
            chunk_size=chunk_size,
            reset_mask=reset_pad,
        )

        if return_last_state:
            h_pad, (c_last, n_last, m_last) = result
            h_seq = h_pad[:, :, :original_seq_len, :]
        else:
            h_pad = result
            h_seq = h_pad[:, :, :original_seq_len, :]
    else:
        result = mlstm_chunkwise__xl_chunk(
            q=queries,
            k=keys,
            v=values,
            i=igate_preact,
            f=fgate_preact,
            c_initial=c_initial,
            n_initial=n_initial,
            m_initial=m_initial,
            return_last_states=return_last_state,
            eps=eps,
            chunk_size=chunk_size,
            reset_mask=reset_mask,
        )

        if return_last_state:
            h_seq, (c_last, n_last, m_last) = result
        else:
            h_seq = result

    if return_last_state:
        if n_last.dim() == 3:
            n_last = n_last.unsqueeze(-1)
        if m_last.dim() == 3:
            m_last = m_last.unsqueeze(-1)
        scale = 1.0 / math.sqrt(hidden)
        return h_seq, (c_last * scale, n_last * scale, m_last)

    return h_seq


__all__ = ["mlstm_chunkwise_triton"]
