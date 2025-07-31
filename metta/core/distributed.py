"""Distributed training utilities for Metta."""

import logging
import os
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def setup_distributed_vars() -> Tuple[bool, int, int]:
    """Set up distributed training variables.

    Returns:
        Tuple of (_master, _world_size, _rank)
    """
    if torch.distributed.is_initialized():
        _master = torch.distributed.get_rank() == 0
        _world_size = torch.distributed.get_world_size()
        _rank = torch.distributed.get_rank()
    else:
        _master = True
        _world_size = 1
        _rank = 0

    return _master, _world_size, _rank


def setup_device_and_distributed(base_device: str = "cuda") -> Tuple[torch.device, bool, int, int]:
    """Set up device and initialize distributed training if needed.

    This function handles:
    - Device selection based on LOCAL_RANK environment variable
    - Distributed process group initialization with appropriate backend
    - Fallback to CPU if CUDA requested but not available
    - Returns distributed training variables (is_master, world_size, rank)

    Args:
        base_device: Base device type ("cuda" or "cpu")

    Returns:
        Tuple of (device, is_master, world_size, rank)
    """
    # Check CUDA availability
    if base_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        base_device = "cpu"

    # Handle distributed setup
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])

        if base_device.startswith("cuda"):
            # CUDA distributed training
            device = torch.device(f"{base_device}:{local_rank}")
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
                logger.info(f"Initialized NCCL distributed training on {device}")
        else:
            # CPU distributed training
            device = torch.device(base_device)
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="gloo")
                logger.info(f"Initialized Gloo distributed training on {device}")
    else:
        # Single device training
        device = torch.device(base_device)
        logger.info(f"Single device training on {device}")

    # Get distributed vars using the shared function
    is_master, world_size, rank = setup_distributed_vars()

    return device, is_master, world_size, rank


def cleanup_distributed() -> None:
    """Destroy the torch distributed process group if initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        logger.info("Destroyed distributed process group")
