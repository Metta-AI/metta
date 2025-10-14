"""Backend selection helpers for kernels.

These helpers decide which backend (CUDA, Triton, or PyTorch) to use based on
runtime conditions. Keep policy separate from generic utils.select_backend.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


def want_cuda_seq_allin(*, tensor: torch.Tensor, seq_len: int, threshold: int) -> bool:
    """Return True if we should prefer the CUDA seq-allin kernel.

    Conditions:
      - tensor is on CUDA device
      - sequence length <= threshold
    """
    return tensor.is_cuda and (seq_len <= threshold)


def load_cuda_stream_diag() -> Optional[Callable]:
    """Try to import the CUDA streaming-diagonal kernel; return callable or None."""
    try:
        from cortex.kernels.cuda import rtu_stream_diag_cuda_seq_allin as cu_fn  # type: ignore

        return cu_fn
    except Exception:
        return None


__all__ = ["want_cuda_seq_allin", "load_cuda_stream_diag"]
