from typing import (
    Optional,
    Tuple,
)

import torch


def _get_cuda_capability() -> Optional[Tuple[int, int]]:
    """Returns the major and minor version of the main device's CUDA capability"""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()


def is_cuda_supported(min_major: int = 7, min_minor: int = 5) -> bool:
    """
    Check if CUDA is available and the primary GPU meets the minimum compute capability.
    According to https://developer.nvidia.com/cuda/gpus, the first non-legacy support starts
    at version 7.5
    """
    capability = _get_cuda_capability()
    if capability is None:
        return False

    major, minor = capability
    if major > min_major:
        return True
    if major == min_major and minor >= min_minor:
        return True

    print(
        f"CUDA available but capability {major}.{minor} < {min_major}.{min_minor}; "
        "test will be skipped or fall back to CPU"
    )
    return False
