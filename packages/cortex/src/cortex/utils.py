"""Utilities for backend selection and Triton availability checks."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Callable

import torch
from torch._dynamo import disable

from cortex.cuda_utils import is_cuda_supported

logger = logging.getLogger(__name__)

_disable_triton_env = os.getenv("CORTEX_DISABLE_TRITON") or os.getenv("CORTEX_FORCE_PYTORCH")
_disable_triton = str(_disable_triton_env).lower() in {"1", "true", "yes"}

if _disable_triton:
    TRITON_AVAILABLE = False
else:
    try:
        import triton  # noqa: F401

        TRITON_AVAILABLE = is_cuda_supported()
    except ImportError:
        TRITON_AVAILABLE = False


def _lazy_import(fn_or_path: Callable | str | None) -> Callable | None:
    """Import a callable from a dotted path if provided."""
    if fn_or_path is None or callable(fn_or_path):
        return fn_or_path

    try:
        module_path, fn_name = fn_or_path.rsplit(":", 1)
        module = importlib.import_module(module_path)
        return getattr(module, fn_name)
    except (ImportError, AttributeError, ValueError) as e:
        logger.debug(f"Failed to import {fn_or_path}: {e}")
        return None


@disable
def select_backend(
    triton_fn: Callable | str | None,
    pytorch_fn: Callable | str,
    tensor: torch.Tensor,
    *,
    allow_triton: bool = True,
    cuda_fn: Callable | str | None = None,
    allow_cuda: bool = False,
) -> Callable:
    """Select CUDA, Triton, or PyTorch backend with lazy loading support."""
    cuda_fn_resolved = _lazy_import(cuda_fn) if (cuda_fn and is_cuda_supported()) else None
    triton_fn_resolved = _lazy_import(triton_fn) if (triton_fn and TRITON_AVAILABLE) else None
    pytorch_fn_resolved = _lazy_import(pytorch_fn)

    if allow_cuda and cuda_fn_resolved is not None and tensor.is_cuda:
        logger.debug(
            "Using CUDA backend for %s (device=%s, dtype=%s)",
            getattr(cuda_fn_resolved, "__name__", "cuda_fn"),
            tensor.device,
            tensor.dtype,
        )
        return cuda_fn_resolved

    use_triton = TRITON_AVAILABLE and triton_fn_resolved is not None and allow_triton and tensor.is_cuda
    if use_triton:
        logger.debug(
            f"Using Triton backend for {triton_fn_resolved.__name__} (device={tensor.device}, dtype={tensor.dtype})"
        )
        return triton_fn_resolved

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
