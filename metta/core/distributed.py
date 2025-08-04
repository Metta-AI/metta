"""Distributed training utilities for Metta."""

import logging
import os
from typing import Tuple

import torch

logger = logging.getLogger(__name__)

# Set critical NCCL environment variables to prevent hangs
# TORCH_NCCL_ASYNC_ERROR_HANDLING=1 allows NCCL to detect and report errors instead of hanging
# This is crucial for preventing the "rank 0 stuck at barrier" issue
if "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in os.environ:
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    logger.info("Set TORCH_NCCL_ASYNC_ERROR_HANDLING=1 to prevent distributed training hangs")

# Also set NCCL_BLOCKING_WAIT to ensure proper synchronization
if "NCCL_BLOCKING_WAIT" not in os.environ:
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    logger.info("Set NCCL_BLOCKING_WAIT=1 for proper barrier synchronization")


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
            # Set the device for this process - critical for distributed training
            torch.cuda.set_device(device)
            if not torch.distributed.is_initialized():
                # Pass device_id to init_process_group to avoid NCCL warnings and potential hangs
                torch.distributed.init_process_group(backend="nccl", device_id=device)
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
        # Ensure all ranks synchronize before destroying process group
        logger.info(f"Rank {torch.distributed.get_rank()}: Entering cleanup barrier")

        # Use monitored_barrier with timeout to handle crashed processes
        from datetime import timedelta

        try:
            torch.distributed.monitored_barrier(timeout=timedelta(seconds=300))  # 5 minute timeout
            logger.info(f"Rank {torch.distributed.get_rank()}: Exited cleanup barrier")
        except Exception as e:
            logger.warning(f"Rank {torch.distributed.get_rank()}: Cleanup barrier timed out or failed: {e}")
            # Continue with cleanup even if barrier fails

        torch.distributed.destroy_process_group()
        logger.info("Destroyed distributed process group")
