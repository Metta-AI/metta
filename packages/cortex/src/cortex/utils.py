"""Utilities for backend selection and Triton availability checks."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Callable

import torch

logger = logging.getLogger(__name__)

# Check if Triton is available and CUDA is available, with an escape hatch
# to force-disable via environment variable (useful for first-run JIT delays
# or troubleshooting kernels).
_disable_triton_env = os.getenv("CORTEX_DISABLE_TRITON") or os.getenv("CORTEX_FORCE_PYTORCH")
_disable_triton = str(_disable_triton_env).lower() in {"1", "true", "yes"}

if _disable_triton:
    TRITON_AVAILABLE = False
else:
    try:
        import triton  # noqa: F401

        TRITON_AVAILABLE = torch.cuda.is_available()
    except ImportError:
        TRITON_AVAILABLE = False


def _lazy_import(fn_or_path: Callable | str | None) -> Callable | None:
    """Import function from string path or return callable as-is.

    Args:
        fn_or_path: Either a callable or a string path like "module.path:function_name"

    Returns:
        The callable, or None if import fails
    """
    if fn_or_path is None or callable(fn_or_path):
        return fn_or_path

    # Parse "module.path:function_name"
    try:
        module_path, fn_name = fn_or_path.rsplit(":", 1)
        module = importlib.import_module(module_path)
        return getattr(module, fn_name)
    except (ImportError, AttributeError, ValueError) as e:
        logger.debug(f"Failed to import {fn_or_path}: {e}")
        return None


def select_backend(
    triton_fn: Callable | str | None,
    pytorch_fn: Callable | str,
    tensor: torch.Tensor,
    *,
    allow_triton: bool = True,
    cuda_fn: Callable | str | None = None,
    allow_cuda: bool = False,
) -> Callable:
    """Select CUDA, Triton, or PyTorch backend with lazy loading support.

    Args:
        triton_fn: Triton function or import path like "module:function"
        pytorch_fn: PyTorch function or import path
        tensor: Input tensor for device/dtype checks
        allow_triton: Whether to allow Triton backend
        cuda_fn: CUDA function or import path
        allow_cuda: Whether to allow CUDA backend

    Returns:
        Selected backend function

    Order of preference:
    1) CUDA (if ``allow_cuda`` and ``cuda_fn`` and tensor on CUDA)
    2) Triton (if available, ``allow_triton``, and tensor on CUDA)
    3) PyTorch (fallback)
    """
    # Lazy import backends only if available
    cuda_fn_resolved = _lazy_import(cuda_fn) if (cuda_fn and torch.cuda.is_available()) else None
    triton_fn_resolved = _lazy_import(triton_fn) if (triton_fn and TRITON_AVAILABLE) else None
    pytorch_fn_resolved = _lazy_import(pytorch_fn)  # PyTorch always available

    # CUDA priority (highest)
    if allow_cuda and cuda_fn_resolved is not None and tensor.is_cuda:
        logger.debug(
            "Using CUDA backend for %s (device=%s, dtype=%s)",
            getattr(cuda_fn_resolved, "__name__", "cuda_fn"),
            tensor.device,
            tensor.dtype,
        )
        return cuda_fn_resolved

    # Triton priority (second)
    use_triton = TRITON_AVAILABLE and triton_fn_resolved is not None and allow_triton and tensor.is_cuda
    if use_triton:
        logger.debug(
            f"Using Triton backend for {triton_fn_resolved.__name__} (device={tensor.device}, dtype={tensor.dtype})"
        )
        return triton_fn_resolved

    # PyTorch fallback (always works)
    reasons = []
    if not TRITON_AVAILABLE:
        reasons.append("Triton not available")
    elif triton_fn_resolved is None:
        reasons.append("no Triton implementation")
    elif not allow_triton:
        reasons.append("Triton not allowed for this call")
    elif not tensor.is_cuda:
        reasons.append(f"tensor on {tensor.device}")

    reason_str = ", ".join(reasons) if reasons else "unknown reason"
    logger.debug(f"Using PyTorch backend for {pytorch_fn_resolved.__name__} ({reason_str})")
    return pytorch_fn_resolved


__all__ = ["TRITON_AVAILABLE", "select_backend"]
