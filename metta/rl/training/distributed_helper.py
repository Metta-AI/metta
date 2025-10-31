"""Helper for distributed training operations."""

import os
from typing import Any, Optional

import torch
import torch.distributed

from metta.agent.policy import DistributedPolicy, Policy
from metta.common.util.log_config import getRankAwareLogger
from metta.rl.system_config import SystemConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import TrainingEnvironmentConfig
from mettagrid.base_config import Config

logger = getRankAwareLogger(__name__)


class TorchDistributedConfig(Config):
    device: str
    is_master: bool
    world_size: int
    rank: int
    local_rank: int
    distributed: bool


class DistributedHelper:
    def __init__(self, system_cfg: SystemConfig):
        assert not torch.distributed.is_initialized(), "Distributed already initialized"

        # Set up PyTorch optimizations (applies to both distributed and non-distributed)
        self._setup_torch_optimizations()

        # Default values for non-distributed case
        config_values = {
            "device": system_cfg.device,
            "is_master": True,
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "distributed": False,
        }

        # Initialize distributed training if conditions are met
        distributed_config = self._setup_distributed_training(system_cfg)
        if distributed_config:
            config_values.update(distributed_config)

        self.config = TorchDistributedConfig(**config_values)

    def _setup_torch_optimizations(self) -> None:
        """Configure PyTorch for optimal performance."""
        # Keep TF32 fast paths enabled on compatible GPUs (using new API)
        if torch.cuda.is_available() and hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.fp32_precision = "tf32"  # type: ignore[attr-defined]
            torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore[attr-defined]
            # Enable SDPA optimizations for better attention performance
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            logger.info("Enabled PyTorch CUDA optimizations")

    def _setup_distributed_training(self, system_cfg: SystemConfig) -> Optional[dict[str, Any]]:
        """Return distributed config values or None if world_size = 1"""
        if "LOCAL_RANK" not in os.environ or torch.device(system_cfg.device).type != "cuda":
            return None

        world_size_str = os.environ.get("WORLD_SIZE") or os.environ.get("NUM_NODES") or "1"
        world_size = int(world_size_str) if world_size_str.strip() else 1

        if world_size <= 1:
            return None

        rank = int(os.environ.get("RANK", os.environ.get("NODE_INDEX", "0")))
        logger.info(f"world_size: {world_size} rank: {rank}")

        torch.distributed.init_process_group(
            backend="nccl",
            timeout=system_cfg.nccl_timeout,
            init_method=os.environ.get("DIST_URL", "env://"),
            world_size=world_size,
            rank=rank,
        )

        torch.cuda.set_device(system_cfg.device)

        distributed_config = {
            "distributed": True,
            "rank": torch.distributed.get_rank(),
            "world_size": torch.distributed.get_world_size(),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "is_master": torch.distributed.get_rank() == 0,
        }

        logger.info(
            f"Initialized distributed training on {system_cfg.device} "
            f"(rank {distributed_config['rank']}/{distributed_config['world_size']})"
        )

        return distributed_config

    @property
    def is_distributed(self) -> bool:
        return self.config.distributed

    def is_master(self) -> bool:
        return self.config.is_master

    def get_world_size(self) -> int:
        return self.config.world_size

    def get_rank(self) -> int:
        return self.config.rank

    def scale_batch_config(
        self,
        trainer_cfg: TrainerConfig,
        env_cfg: TrainingEnvironmentConfig,
    ) -> None:
        """Scale batch sizes for distributed training if configured.

        When scale_batches_by_world_size is True, this divides batch sizes
        by the world size to maintain consistent global batch size across
        different numbers of GPUs.

        Args:
            trainer_cfg: Trainer configuration to modify in-place
            env_cfg: Optional environment configuration to modify
        """
        if not self.is_distributed or not trainer_cfg.scale_batches_by_world_size:
            return

        # Scale batch sizes by world size
        trainer_cfg.batch_size = trainer_cfg.batch_size // self.get_world_size()

        # Scale forward pass minibatch size
        env_cfg.forward_pass_minibatch_target_size = max(
            1, env_cfg.forward_pass_minibatch_target_size // self.get_world_size()
        )

        logger.info(
            "Scaled batch config for %s processes: batch_size=%s, forward_pass_minibatch_target_size=%s",
            self.get_world_size(),
            trainer_cfg.batch_size,
            env_cfg.forward_pass_minibatch_target_size
            if env_cfg is not None
            else getattr(trainer_cfg, "forward_pass_minibatch_target_size", "n/a"),
        )

    def wrap_policy(self, policy: Policy, device: Optional[torch.device] = None) -> Policy | DistributedPolicy:
        """Wrap policy for distributed training if needed.

        Args:
            policy: Policy to wrap
            device: Device to use (defaults to self.config.device)

        Returns:
            Wrapped policy if distributed, original otherwise
        """
        if not self.is_distributed:
            return policy

        if device is None:
            device = torch.device(self.config.device)

        distributed_policy = DistributedPolicy(policy, device)
        logger.info(f"Wrapped policy with DDP on rank {self.get_rank()}")
        return distributed_policy

    def synchronize(self) -> None:
        """Synchronize across all distributed processes."""
        if self.is_distributed:
            torch.distributed.barrier()

    def broadcast_from_master(self, obj: Any) -> Any:
        """Broadcast object from master to all processes."""
        if not self.is_distributed:
            return obj

        objects = [obj if self.is_master else None]
        torch.distributed.broadcast_object_list(objects, src=0)
        return objects[0]

    def should_log(self) -> bool:
        """Check if this process should perform logging."""
        return self.is_master()

    def should_checkpoint(self) -> bool:
        """Check if this process should save checkpoints."""
        return self.is_master()

    def should_evaluate(self) -> bool:
        """Check if this process should run evaluation."""
        return self.is_master()

    def all_gather(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Gather tensors from all processes."""
        if not self.is_distributed:
            return [tensor]

        gathered = [torch.zeros_like(tensor) for _ in range(self.get_world_size())]
        torch.distributed.all_gather(gathered, tensor)
        return gathered

    def all_reduce(self, tensor: torch.Tensor, op: Any = torch.distributed.ReduceOp.SUM) -> torch.Tensor:
        """Reduce tensor across all processes.

        Args:
            tensor: Tensor to reduce
            op: Reduction operation (default: SUM)

        Returns:
            Reduced tensor (in-place operation)
        """
        if self.is_distributed:
            torch.distributed.all_reduce(tensor, op=op)
        return tensor

    def cleanup(self) -> None:
        """Destroy the torch distributed process group if initialized."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Destroyed distributed process group")
