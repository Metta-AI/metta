"""Utilities for backend selection and Triton availability checks."""

from __future__ import annotations

import logging
from typing import Callable

import torch

logger = logging.getLogger(__name__)

# Check if Triton is available and CUDA is available
try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TRITON_AVAILABLE = False


def select_backend(
    triton_fn: Callable | None,
    pytorch_fn: Callable,
    tensor: torch.Tensor,
    *,
    allow_triton: bool = True,
) -> Callable:
    """Select backend at runtime based on tensor device and conditions.

    Args:
        triton_fn: Triton implementation function (or None if not available)
        pytorch_fn: PyTorch reference implementation function
        tensor: Input tensor to check device
        allow_triton: Whether Triton is allowed for this call (e.g., False for step mode)

    Returns:
        Selected backend function (Triton or PyTorch)
    """
    use_triton = TRITON_AVAILABLE and triton_fn is not None and allow_triton and tensor.is_cuda

    if use_triton:
        logger.debug(f"Using Triton backend for {triton_fn.__name__} (device={tensor.device}, dtype={tensor.dtype})")
        return triton_fn  # type: ignore[return-value]
    else:
        reasons = []
        if not TRITON_AVAILABLE:
            reasons.append("Triton not available")
        elif triton_fn is None:
            reasons.append("no Triton implementation")
        elif not allow_triton:
            reasons.append("Triton not allowed for this call")
        elif not tensor.is_cuda:
            reasons.append(f"tensor on {tensor.device}")

        reason_str = ", ".join(reasons) if reasons else "unknown reason"
        logger.debug(f"Using PyTorch backend for {pytorch_fn.__name__} ({reason_str})")
        return pytorch_fn


__all__ = ["TRITON_AVAILABLE", "select_backend"]
