"""Distributed training setup for Metta."""

import logging
import os
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def setup_distributed_training(base_device: str = "cuda") -> Tuple[torch.device, bool, int, int]:
    """Set up device and distributed training, returning all needed information.

    This combines device setup and distributed initialization into a single call,
    matching the initialization pattern from tools/train.py.

    Args:
        base_device: Base device string ("cuda" or "cpu")

    Returns:
        Tuple of (device, is_master, world_size, rank)
        - device: The torch.device to use for training
        - is_master: True if this is the master process (rank 0)
        - world_size: Total number of processes (1 if not distributed)
        - rank: Current process rank (0 if not distributed)
    """
    # Check if we're in a distributed environment
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])

        # For CUDA, use device with local rank
        if base_device.startswith("cuda"):
            device = torch.device(f"{base_device}:{local_rank}")
            backend = "nccl"
        else:
            # For CPU, just use cpu device (no rank suffix)
            device = torch.device(base_device)
            backend = "gloo"

        torch.distributed.init_process_group(backend=backend)

        # Get distributed info after initialization
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        is_master = rank == 0
    else:
        # Single GPU or CPU
        device = torch.device(base_device)
        rank = 0
        world_size = 1
        is_master = True

    logger.info(f"Using device: {device} (rank {rank}/{world_size})")
    return device, is_master, world_size, rank


def cleanup_distributed():
    """Clean up distributed training if it was initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
