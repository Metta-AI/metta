"""Helper for distributed training operations."""

import logging
import os
from typing import TYPE_CHECKING, Any, List

import torch
import torch.distributed

from metta.agent.policy import Policy
from metta.mettagrid.config import Config

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig

logger = logging.getLogger(__name__)


class TorchDistributedConfig(Config):
    device: str
    is_master: bool
    world_size: int
    rank: int
    local_rank: int
    distributed: bool


class DistributedHelper:
    """Helper class for distributed training operations."""

    def __init__(self, device: torch.device):
        """Initialize distributed helper.

        Args:
            device: Device string (e.g. "cuda:0", "cpu")
        """
        assert not torch.distributed.is_initialized()

        master = True
        world_size = 1
        rank = 0
        local_rank = 0
        distributed = False

        if "LOCAL_RANK" in os.environ and device.type == "cuda":
            torch.distributed.init_process_group(backend="nccl")

            torch.cuda.set_device(device)
            distributed = True
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            master = rank == 0
            logger.info(f"Initialized NCCL distributed training on {device.type}")

        self._config = TorchDistributedConfig(
            device=device.type,
            is_master=master,
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            distributed=distributed,
        )
        self._is_distributed = distributed
        self._is_master = master
        self._rank = rank
        self._world_size = world_size

    def setup(self) -> None:
        """Initialize distributed training if needed."""
        if self._is_distributed:
            logger.info(f"Setting up distributed training for rank {self._rank}")

    def scale_batch_config(self, trainer_cfg: TrainerConfig) -> None:
        """Scale batch sizes for distributed training if configured.

        When scale_batches_by_world_size is True, this divides batch sizes
        by the world size to maintain consistent global batch size across
        different numbers of GPUs.

        Args:
            trainer_cfg: Trainer configuration to modify in-place
        """
        if not self._is_distributed:
            return

        if not trainer_cfg.scale_batches_by_world_size:
            return

        # Scale batch sizes by world size
        trainer_cfg.forward_pass_minibatch_target_size = (
            trainer_cfg.forward_pass_minibatch_target_size // self._world_size
        )
        trainer_cfg.batch_size = trainer_cfg.batch_size // self._world_size

        logger.info(
            f"Scaled batch config for {self._world_size} processes: "
            f"batch_size={trainer_cfg.batch_size}, "
            f"forward_pass_minibatch_target_size={trainer_cfg.forward_pass_minibatch_target_size}"
        )

    def wrap_policy(self, policy: Policy, device: torch.device) -> Policy:
        """Wrap policy for distributed training if needed.

        Args:
            policy: Policy to wrap
            device: Device to use

        Returns:
            Wrapped policy if distributed, original otherwise
        """
        if self._is_distributed:
            # Wrap with DDP
            policy = torch.nn.parallel.DistributedDataParallel(
                policy,
                device_ids=[device.index] if device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            logger.info(f"Wrapped policy with DDP on rank {self._rank}")
        return policy

    def synchronize(self) -> None:
        """Synchronize across processes if distributed."""
        if self._is_distributed:
            torch.distributed.barrier()

    def broadcast_from_master(self, obj: Any) -> Any:
        """Broadcast object from master to all processes.

        Args:
            obj: Object to broadcast

        Returns:
            Broadcasted object
        """
        if not self._is_distributed:
            return obj

        # Create a list to hold the object on all ranks
        objects = [obj if self._is_master else None]
        torch.distributed.broadcast_object_list(objects, src=0)
        return objects[0]

    def is_master(self) -> bool:
        """Check if this is the master process.

        Returns:
            True if master process
        """
        return self._is_master

    def should_log(self) -> bool:
        """Check if this process should perform logging.

        Returns:
            True if should log
        """
        return self._is_master

    def should_checkpoint(self) -> bool:
        """Check if this process should save checkpoints.

        Returns:
            True if should checkpoint
        """
        return self._is_master

    def should_evaluate(self) -> bool:
        """Check if this process should run evaluation.

        Returns:
            True if should evaluate
        """
        return self._is_master

    def get_world_size(self) -> int:
        """Get the number of processes.

        Returns:
            World size
        """
        return self._world_size

    def get_rank(self) -> int:
        """Get the rank of this process.

        Returns:
            Process rank
        """
        return self._rank

    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensors from all processes.

        Args:
            tensor: Tensor to gather

        Returns:
            List of tensors from all processes
        """
        if not self._is_distributed:
            return [tensor]

        gathered = [torch.zeros_like(tensor) for _ in range(self._world_size)]
        torch.distributed.all_gather(gathered, tensor)
        return gathered

    def all_reduce(
        self, tensor: torch.Tensor, op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM
    ) -> torch.Tensor:
        """Reduce tensor across all processes.

        Args:
            tensor: Tensor to reduce
            op: Reduction operation

        Returns:
            Reduced tensor
        """
        if self._is_distributed:
            torch.distributed.all_reduce(tensor, op=op)
        return tensor

    def cleanup(self) -> None:
        """Destroy the torch distributed process group if initialized."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Destroyed distributed process group")
