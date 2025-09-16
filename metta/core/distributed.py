"""Distributed training utilities for Metta."""

import logging
import os
from datetime import timedelta

import torch

from metta.mettagrid.config import Config

logger = logging.getLogger(__name__)


class TorchDistributedConfig(Config):
    device: str
    is_master: bool
    world_size: int
    rank: int
    local_rank: int
    distributed: bool


def _init_process_group() -> bool:
    world_size_str = os.environ.get("WORLD_SIZE") or os.environ.get("NUM_NODES") or "1"
    world_size = int(world_size_str) if world_size_str.strip() else 1
    if world_size <= 1:
        return False
    if torch.distributed.is_initialized():
        return False

    rank = int(os.environ.get("RANK", os.environ.get("NODE_INDEX", "0")))
    torch.distributed.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=30),
        init_method=os.environ.get("DIST_URL", "env://"),
        world_size=world_size,
        rank=rank,
    )
    return True


def enable_nccl_watchdog():
    """
    Enable (and reset) the NCCL watchdog
    """
    # Enable watchdog
    os.environ["NCCL_WATCHDOG_ENABLE"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

    # Setting these environment variables effectively resets the timer
    # as NCCL will read them on the next collective operation

    logger.warning("NCCL watchdog enabled!")


def disable_nccl_watchdog():
    """
    Disable the NCCL watchdog.
    """
    # Disable watchdog
    os.environ["NCCL_WATCHDOG_ENABLE"] = "0"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"

    logger.warning("NCCL watchdog disabled")


def setup_torch_distributed(device: str) -> TorchDistributedConfig:
    assert not torch.distributed.is_initialized()

    master = True
    world_size = 1
    rank = 0
    local_rank = 0
    distributed = False

    if "LOCAL_RANK" in os.environ and device.startswith("cuda"):
        _init_process_group()

        torch.cuda.set_device(device)
        distributed = True
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        master = rank == 0
        logger.info(f"Initialized NCCL distributed training on {device}")

    return TorchDistributedConfig(
        device=device,
        is_master=master,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed=distributed,
    )


def cleanup_distributed() -> None:
    """Destroy the torch distributed process group if initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        logger.info("Destroyed distributed process group")
