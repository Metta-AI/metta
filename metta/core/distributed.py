"""Distributed training utilities for Metta."""

import logging
import os

import torch

<<<<<<< HEAD
from metta.core.distributed_config import DistributedConfig
=======
from metta.common.config import Config
>>>>>>> refs/remotes/origin/main

logger = logging.getLogger(__name__)


<<<<<<< HEAD
def setup_distributed_vars() -> DistributedConfig:
    """Set up distributed training variables.

    Returns:
        DistributedConfig with master, world_size, and rank information
    """
    if torch.distributed.is_initialized():
        _world_size = torch.distributed.get_world_size()
        _rank = torch.distributed.get_rank()
        _master = _rank == 0
    else:
        _master = True
        _world_size = 1
        _rank = 0

    return DistributedConfig(world_size=_world_size, rank=_rank, is_master=_master)


def setup_device_and_distributed(base_device: str = "cuda") -> Tuple[torch.device, DistributedConfig]:
    """Set up device and initialize distributed training if needed.
=======
class TorchDistributedConfig(Config):
    device: str
    is_master: bool
    world_size: int
    rank: int
    local_rank: int
    distributed: bool


def setup_torch_distributed(device: str) -> TorchDistributedConfig:
    assert not torch.distributed.is_initialized()
>>>>>>> refs/remotes/origin/main

    master = True
    world_size = 1
    rank = 0
    local_rank = 0
    distributed = False

    if "LOCAL_RANK" in os.environ and device.startswith("cuda"):
        torch.distributed.init_process_group(backend="nccl")

<<<<<<< HEAD
    Returns:
        Tuple of (device, distributed_config)
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

            # Set the device for the current process
            # this prevents problems with collective operations that happen before the policy is wrapped
            logger.info(f"Setting device to {device}")
            torch.cuda.set_device(device)
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
    distributed_config = setup_distributed_vars()

    return device, distributed_config
=======
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
>>>>>>> refs/remotes/origin/main


def cleanup_distributed() -> None:
    """Destroy the torch distributed process group if initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        logger.info("Destroyed distributed process group")
