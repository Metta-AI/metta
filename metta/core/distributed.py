"""Distributed training utilities for Metta."""

import logging
import os

import torch

from metta.common.config import Config

logger = logging.getLogger(__name__)


class TorchDistributedConfig(Config):
    device: str
    is_master: bool
    world_size: int
    rank: int
    local_rank: int
    distributed: bool


def setup_torch_distributed(device: str) -> TorchDistributedConfig:
    assert not torch.distributed.is_initialized()

    master = True
    world_size = 1
    rank = 0
    local_rank = 0
    distributed = False

    if "LOCAL_RANK" in os.environ and device.startswith("cuda"):
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl")

        # Set device using local_rank integer to avoid parsing issues
        torch.cuda.set_device(local_rank)
        # Update device string to match what we actually set
        device = f"cuda:{local_rank}"
        distributed = True
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        master = rank == 0
        logger.info(f"Initialized NCCL distributed training on {device} (rank {rank}, local_rank {local_rank})")

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
