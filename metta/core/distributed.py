"""Distributed training utilities for Metta."""

import logging
import os

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


def setup_torch_distributed(device: str) -> TorchDistributedConfig:
    assert not torch.distributed.is_initialized()

    master = True
    world_size = 1
    rank = 0
    local_rank = 0
    distributed = False

    if "LOCAL_RANK" in os.environ and device.startswith("cuda"):
        torch.distributed.init_process_group(backend="nccl")

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
