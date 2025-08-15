"""Distributed training utilities for Metta."""

import logging

import torch

from metta.common.util.config import Config

logger = logging.getLogger(__name__)


class TorchDistributedConfig(Config):
    device: str
    is_master: bool
    world_size: int
    rank: int
    local_rank: int


def setup_torch_distributed(device: str) -> TorchDistributedConfig:
    master = True
    world_size = 1
    rank = 0
    local_rank = 0

    if device.startswith("cuda"):
        # CUDA distributed training
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
            logger.info(f"Initialized NCCL distributed training on {device}")

        torch.cuda.set_device(device)
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        master = rank == 0

    return TorchDistributedConfig(
        device=device,
        is_master=master,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
    )


def cleanup_distributed() -> None:
    """Destroy the torch distributed process group if initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        logger.info("Destroyed distributed process group")
