"""Centralized PyTorch initialization for the Metta codebase.

This module configures PyTorch settings (like TF32 precision) that should
be set once globally before any models are created or compiled.
"""

import os

import torch
from cortex.torch_init import enable_determinism, seed_everything, set_tf32_precision

# Flag to ensure we only configure once
_configured = False


def configure_torch_globally_for_performance() -> None:
    """
    Configure PyTorch settings globally (TF32, etc.) for performance.

    This should be called early in the application lifecycle, before any
    models are created or torch.compile is called. It's safe to call
    multiple times (idempotent).
    """
    global _configured
    if _configured:
        return

    # Configure TF32 precision for CUDA performance.
    set_tf32_precision(True)

    # Enable CuDNN benchmark mode for performance.
    torch.backends.cudnn.benchmark = True

    _configured = True


def seed_everything_distributed_aware(base_seed: int, /) -> None:
    # Add rank offset to base seed for distributed training to ensure different
    # processes generate uncorrelated random sequences
    rank = int(os.environ.get("RANK", 0))
    rank_specific_seed = base_seed + rank

    seed_everything(rank_specific_seed)


def configure_torch_for_determinism() -> None:
    """
    Configure PyTorch settings globally (TF32, etc.) for determinism.
    """
    enable_determinism()
