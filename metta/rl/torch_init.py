"""Centralized PyTorch initialization for the Metta codebase.

This module configures PyTorch settings (like TF32 precision) that should
be set once globally before any models are created or compiled.
"""

import os
import random

import numpy as np
import torch


def set_tf32_precision(enabled: bool, /) -> None:
    if not torch.cuda.is_available():
        return

    # For now, we ALWAYS use the legacy allow_tf32 API as torch._inductor appears to have an issue with the newer
    #  fp32 tf32 API: https://github.com/pytorch/pytorch/issues/166387
    # Trying to only use the new API led to a "mix of legacy and new APIs" RuntimeError() on compilation of
    #  certain Cortex cells.
    # When PyTorch removes support for legacy API, we can move to the new API, and hopefully, the bug is fixed by then.
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = enabled
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = enabled

    # enabled = mode if isinstance(mode, bool) else mode.lower() == "tf32"
    # matmul_has_fp32 = hasattr(torch.backends.cuda.matmul, "fp32_precision")
    # cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
    # cudnn_has_fp32 = cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision")
    #
    # if matmul_has_fp32:
    #     torch.backends.cuda.matmul.fp32_precision = "tf32" if enabled else "ieee"
    # if cudnn_has_fp32:
    #     cudnn_conv.fp32_precision = "tf32" if enabled else "ieee"
    #
    # if not matmul_has_fp32 and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
    #     torch.backends.cuda.matmul.allow_tf32 = enabled
    # if not cudnn_has_fp32 and hasattr(torch.backends.cudnn, "allow_tf32"):
    #     torch.backends.cudnn.allow_tf32 = enabled


# Flag to ensure we only configure once
_configured = False


def configure_torch_globally_for_performance() -> None:
    """Configure PyTorch settings globally (TF32, etc.) for performance.

    This should be called early in the application lifecycle, before any
    models are created or torch.compile is called. It's safe to call
    multiple times (idempotent).
    """
    global _configured
    if _configured:
        return

    # Configure TF32 precision for CUDA (performance mode)
    set_tf32_precision(True)

    _configured = True


def seed_everything(base_seed: int, /) -> None:
    # Add rank offset to base seed for distributed training to ensure different
    # processes generate uncorrelated random sequences
    rank = int(os.environ.get("RANK", 0))
    rank_specific_seed = base_seed + rank

    random.seed(rank_specific_seed)
    np.random.seed(rank_specific_seed)
    torch.manual_seed(rank_specific_seed)
    torch.cuda.manual_seed_all(rank_specific_seed)


def enable_determinism() -> None:
    """Enable deterministic behavior (overrides performance settings).

    This disables TF32 and sets other deterministic flags.
    Should be called when reproducibility is more important than performance.
    """
    # Set CuBLAS workspace config for deterministic behavior on CUDA >= 10.2
    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    set_tf32_precision(False)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Despite above efforts, we still don't get deterministic behavior.
#  But presumably this is better than nothing.
#  https://docs.pytorch.org/docs/stable/notes/randomness.html#reproducibility
