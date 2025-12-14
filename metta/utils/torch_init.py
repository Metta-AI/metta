"""Centralized PyTorch initialization for the Metta codebase.

This module configures PyTorch settings (like TF32 precision) that should
be set once globally before any models are created or compiled.
"""

import os

import torch

# Flag to ensure we only configure once
_configured = False
_deterministic_mode = False


def configure_torch_globally(deterministic: bool = False) -> None:
    """Configure PyTorch settings globally (TF32, etc.).

    This should be called early in the application lifecycle, before any
    models are created or torch.compile is called. It's safe to call
    multiple times (idempotent).

    Args:
        deterministic: If True, sets highest precision (disables TF32) for
            deterministic behavior. If False, sets "high" for performance.
    """
    global _configured, _deterministic_mode

    # If already configured with same mode, skip
    if _configured and _deterministic_mode == deterministic:
        return

    # Configure TF32 precision for CUDA
    if torch.cuda.is_available():
        if deterministic:
            # "highest" precision disables TF32 (for deterministic behavior)
            torch.set_float32_matmul_precision("highest")
        else:
            # "high" precision enables TF32 (for performance)
            torch.set_float32_matmul_precision("high")

    _configured = True
    _deterministic_mode = deterministic


def enable_determinism() -> None:
    """Enable deterministic behavior (overrides performance settings).

    This sets TF32 to "highest" precision and other deterministic flags.
    Should be called when reproducibility is more important than performance.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    configure_torch_globally(deterministic=True)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


# Auto-configure on import (runs once when module is first imported)
# Check for deterministic mode via environment variable
_deterministic_env = os.getenv("PYTORCH_DETERMINISTIC", "").lower() in ("1", "true", "yes")
configure_torch_globally(deterministic=_deterministic_env)

