from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = torch.cuda.is_available()
except Exception:  # pragma: no cover - import guard
    TRITON_AVAILABLE = False


def slstm_sequence_triton(
    *,
    Wx: torch.Tensor,  # (B, T, 4, NH, DH) feed-forward preactivations (i, f, z, o)
    R: torch.Tensor,  # (4, NH, DH, DH) recurrent weights per gate
    b: torch.Tensor,  # (4, NH, DH) bias per gate
    initial_states: torch.Tensor,  # (4, B, NH, DH) states (h, c, n, m)
    autocast_kernel_dtype: str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run sLSTM sequence using Triton kernel.

    Returns a tuple of:
      - all_states: (T, 4, B, NH, DH)
      - last_state: (4, B, NH, DH)
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available or CUDA not enabled")

    from .slstm_triton.torch import slstm_tr_fwbw

    assert Wx.dim() == 5 and Wx.shape[2] == 4, f"Wx must be (B,T,4,NH,DH), got {Wx.shape}"
    assert R.shape[0] == 4, f"R must be (4,NH,DH,DH), got {R.shape}"
    assert b.shape[0] == 4, f"b must be (4,NH,DH), got {b.shape}"
    assert initial_states.shape[0] == 4, f"initial_states must be (4,B,NH,DH), got {initial_states.shape}"

    # Select kernel dtype policy
    if autocast_kernel_dtype is None:
        if Wx.dtype == torch.bfloat16:
            autocast_kernel_dtype = "bfloat16"
        elif Wx.dtype == torch.float16:
            autocast_kernel_dtype = "float16"
        else:
            autocast_kernel_dtype = "float32"

    all_states, last_state = slstm_tr_fwbw(
        states_initial=initial_states,
        Wx=Wx,
        R=R,
        b=b,
        autocast_kernel_dtype=autocast_kernel_dtype,
    )
    return all_states, last_state


__all__ = ["TRITON_AVAILABLE", "slstm_sequence_triton"]
