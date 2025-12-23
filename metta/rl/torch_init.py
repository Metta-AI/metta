"""Centralized PyTorch initialization for the Metta codebase.

This module configures PyTorch settings (like TF32 precision) that should
be set once globally before any models are created or compiled.
"""

import os

import torch

# Flag to ensure we only configure once
_configured = False


def _set_tf32_precision(enabled: bool) -> None:
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32" if enabled else "ieee"
        cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
        if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
            cudnn_conv.fp32_precision = "tf32" if enabled else "ieee"
        return

    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = enabled
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = enabled


def configure_torch_globally() -> None:
    """Configure PyTorch settings globally (TF32, etc.) for performance.

    This should be called early in the application lifecycle, before any
    models are created or torch.compile is called. It's safe to call
    multiple times (idempotent).
    """
    global _configured

    if _configured:
        return

    # Configure TF32 precision for CUDA (performance mode)
    if torch.cuda.is_available():
        _set_tf32_precision(True)

    _configured = True


def enable_determinism() -> None:
    """Enable deterministic behavior (overrides performance settings).

    This disables TF32 and sets other deterministic flags.
    Should be called when reproducibility is more important than performance.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        _set_tf32_precision(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Auto-configure on import (runs once when module is first imported)
configure_torch_globally()
