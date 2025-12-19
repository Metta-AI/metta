"""Triton sLSTM kernels package exports."""

from __future__ import annotations

from typing import Optional

import torch as pt
from cortex.kernels.triton.slstm.torch.fwbw import slstm_tr_fwbw


def slstm_sequence_triton(
    *,
    Wx: pt.Tensor,  # (B, T, NGI=4, NH, DH)
    R: pt.Tensor,  # (NGR=4, NH, DH, DH)
    b: pt.Tensor,  # (NGI=4, NH, DH)
    initial_states: pt.Tensor,  # (NS=4, B, NH, DH)
    resets: Optional[pt.Tensor] = None,  # (B, T)
) -> tuple[pt.Tensor, pt.Tensor]:
    autocast_kernel_dtype: str
    if Wx.dtype == pt.float32:
        autocast_kernel_dtype = "float32"
    elif Wx.dtype == pt.float16:
        autocast_kernel_dtype = "float16"
    elif Wx.dtype == pt.bfloat16:
        autocast_kernel_dtype = "bfloat16"
    else:
        raise ValueError(f"Unsupported dtype for sLSTM Triton kernel: {Wx.dtype}")

    all_states, last_state = slstm_tr_fwbw(
        states_initial=initial_states,
        Wx=Wx,
        R=R,
        b=b,
        resets=resets,
        autocast_kernel_dtype=autocast_kernel_dtype,
    )
    return all_states, last_state


__all__ = ["slstm_sequence_triton"]
