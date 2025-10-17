#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

"""Minimal triton kernels for mLSTM from mlstm_kernels package."""

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
    reset_mask: Optional[pt.Tensor] = None,  # B, S - boolean mask for resets
    chunk_size: int = 64,
    return_last_state: bool = False,
    eps: float = 1e-6,
    **kwargs,
) -> tuple[pt.Tensor, tuple[pt.Tensor, pt.Tensor, pt.Tensor]]:
    """Triton-accelerated chunkwise mLSTM implementation.

    Uses optimized Triton kernels when available via lazy import.
    Falls back to simple implementation if triton is not available.

    Args:
        queries: (B, NH, S, DH)
        keys: (B, NH, S, DH)
        values: (B, NH, S, DH)
        igate_preact: (B, NH, S)
        fgate_preact: (B, NH, S)
        initial_C: (B, NH, DH, DH), optional
        initial_n: (B, NH, DH) or (B, NH, DH, 1), optional
        initial_m: (B, NH, 1, 1), optional
        chunk_size: Size of chunks for processing
        return_last_state: Whether to return final states
        eps: Small constant for numerical stability

    Returns:
        Output tensor (B, NH, S, DH) and optionally final states (C, n, m)
    """
    # Lazy import to avoid loading unnecessary modules
    try:
        from cortex.kernels.triton.mlstm.torch import mlstm_chunkwise__xl_chunk
    except ImportError:
        # Fallback if Triton kernels are not available
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

    # Use Triton kernel in a single call; internal kernels handle resets and chunking
    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    c_initial = initial_C if initial_C is not None else pt.zeros((B, NH, DH, DH), dtype=_dtype, device=_device)
    n_initial = initial_n if initial_n is not None else pt.zeros((B, NH, DH), dtype=_dtype, device=_device)
    if n_initial.dim() == 4:
        n_initial = n_initial.squeeze(-1)
    m_initial = initial_m if initial_m is not None else pt.zeros((B, NH, 1), dtype=_dtype, device=_device)
    if m_initial.dim() == 4:
        m_initial = m_initial.squeeze(-1)

    # Transparent padding to multiple of 16 with identity steps
    S_orig = S
    pad = (16 - (S_orig % 16)) % 16
    if pad > 0:
        zeros_t = pt.zeros(B, NH, pad, DH, dtype=_dtype, device=_device)
        q_pad = pt.cat([queries, zeros_t], dim=2)
        k_pad = pt.cat([keys, zeros_t], dim=2)
        v_pad = pt.cat([values, zeros_t], dim=2)

        # For padded timesteps: forget≈1, input≈0 so state is unchanged
        if _dtype in (pt.float16, pt.bfloat16):
            pos = pt.tensor(10.0, dtype=_dtype, device=_device)
        else:
            pos = pt.tensor(20.0, dtype=_dtype, device=_device)
        neg = -pos
        i_tail = neg.expand(B, NH, pad)
        f_tail = pos.expand(B, NH, pad)
        i_pad = pt.cat([igate_preact, i_tail], dim=2)
        f_pad = pt.cat([fgate_preact, f_tail], dim=2)

        rm_pad = None
        if reset_mask is not None:
            rm_tail = pt.zeros(B, pad, dtype=reset_mask.dtype, device=_device)
            rm_pad = pt.cat([reset_mask, rm_tail], dim=1)

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
            reset_mask=rm_pad,
        )

        if return_last_state:
            h_pad, (c_last, n_last, m_last) = result
            h_seq = h_pad[:, :, :S_orig, :]
        else:
            h_pad = result
            h_seq = h_pad[:, :, :S_orig, :]
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
        scale = 1.0 / math.sqrt(DH)
        return h_seq, (c_last * scale, n_last * scale, m_last)
    else:
        return h_seq


__all__ = ["mlstm_chunkwise_triton"]
