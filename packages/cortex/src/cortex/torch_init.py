"""
PyTorch initialization utilities for Cortex.

This module contains functions for configuring PyTorch settings like TF32 precision,
seeding, and deterministic behavior.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from cortex.cuda_utils import is_cuda_supported


def set_tf32_precision(enabled: bool, /) -> None:
    if not is_cuda_supported():
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


def seed_everything(seed: int, /) -> None:
    """
    Seed everything for reproducibility.

    If calling this from a distributed training context or from within metta code,
    use `seed_everything_distributed_aware()` instead.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_determinism() -> None:
    """
    Enable deterministic behavior (overrides performance settings).

    This disables TF32 and sets other determinism flags.
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
