"""Triton SLSTM kernel implementation."""

from __future__ import annotations

from typing import Tuple

import torch as pt


def slstm_sequence_triton(
    *,
    Wx: pt.Tensor,  # (B, T, 4, NH, DH) feed-forward preactivations (i, f, z, o)
    R: pt.Tensor,  # (4, NH, DH, DH) recurrent weights per gate
    b: pt.Tensor,  # (4, NH, DH) bias per gate
    initial_states: pt.Tensor,  # (4, B, NH, DH) states (h, c, n, m)
    resets: pt.Tensor | None = None,  # (B, T) reset mask applied before each timestep
    autocast_kernel_dtype: str | None = None,
) -> Tuple[pt.Tensor, pt.Tensor]:
    """Run sLSTM sequence using Triton kernel.

    Args:
        Wx: (B, T, 4, NH, DH) feed-forward preactivations in order (i, f, z, o)
        R: (4, NH, DH, DH) recurrent weights per gate in order (i, f, z, o)
        b: (4, NH, DH) bias per gate in order (i, f, z, o)
        initial_states: (4, B, NH, DH) states (h, c, n, m)
        resets: (B, T) reset mask, optional. When provided, states are zeroed for
            entries where resets[:, t] is True prior to processing timestep t.
        autocast_kernel_dtype: dtype for kernel computation

    Returns:
        all_states: (T, 4, B, NH, DH)
        last_state: (4, B, NH, DH)
    """
    from cortex.kernels.triton.slstm.torch import slstm_tr_fwbw

    assert Wx.dim() == 5 and Wx.shape[2] == 4, f"Wx must be (B,T,4,NH,DH), got {Wx.shape}"
    assert R.shape[0] == 4, f"R must be (4,NH,DH,DH), got {R.shape}"
    assert b.shape[0] == 4, f"b must be (4,NH,DH), got {b.shape}"
    assert initial_states.shape[0] == 4, f"initial_states must be (4,B,NH,DH), got {initial_states.shape}"

    # Force fp32 compute inside the Triton kernels unless explicitly overridden
    if autocast_kernel_dtype is None:
        autocast_kernel_dtype = "float32"

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
