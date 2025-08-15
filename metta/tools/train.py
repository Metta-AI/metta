import copy
import logging
import os
from logging import Logger
from typing import Any, Optional

import torch
from omegaconf import OmegaConf
from pydantic import Field

from metta.agent.policy_store import PolicyStore
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.util.git import get_git_hash_for_remote_task
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.logging_helpers import init_file_logging, init_logging
from metta.common.util.tool import Tool
from metta.common.wandb.wandb_context import WandbConfig, WandbConfigOff, WandbContext, WandbRun
from metta.core.distributed import TorchDistributedConfig, setup_torch_distributed
from metta.rl.system_config import SystemConfig
from metta.rl.trainer import train
from metta.rl.trainer_config import TrainerConfig

logger = logging.getLogger(__name__)


class TrainTool(Tool):
    system: SystemConfig = Field(default_factory=SystemConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    wandb: WandbConfig = WandbConfigOff()
    policy_architecture: Any
    run: str
    run_dir: Optional[str] = None

    # Stats server configuration
    stats_server_uri: Optional[str] = "https://api.observatory.softmax-research.net"

    # Policy configuration
    policy_uri: Optional[str] = None

    # Optional configurations
    map_preview_uri: str | None = None

    def model_post_init(self, __context):
        # Set run_dir based on run name if not explicitly set
        if self.run_dir is None:
            self.run_dir = f"{self.system.data_dir}/{self.run}"

        # Set policy_uri if not set
        if not self.policy_uri:
            self.policy_uri = f"file://{self.run_dir}/checkpoints"

        # Set up checkpoint and replay directories
        if not self.trainer.checkpoint.checkpoint_dir:
            self.trainer.checkpoint.checkpoint_dir = f"{self.run_dir}/checkpoints/"
        if self.trainer.evaluation and not self.trainer.evaluation.replay_dir:
            self.trainer.evaluation.replay_dir = f"{self.run_dir}/replays/"

    def to_mini(self) -> "TrainTool":
        cfg = copy.deepcopy(self)
        cfg.trainer.minibatch_size = min(cfg.trainer.minibatch_size, 1024)
        cfg.trainer.batch_size = min(cfg.trainer.batch_size, 1024)
        cfg.trainer.async_factor = 1
        cfg.trainer.forward_pass_minibatch_target_size = min(cfg.trainer.forward_pass_minibatch_target_size, 4)
        cfg.trainer.checkpoint.checkpoint_interval = min(cfg.trainer.checkpoint.checkpoint_interval, 10)
        cfg.trainer.checkpoint.wandb_checkpoint_interval = min(cfg.trainer.checkpoint.wandb_checkpoint_interval, 10)
        cfg.trainer.bptt_horizon = min(cfg.trainer.bptt_horizon, 8)
        if cfg.trainer.evaluation:
            cfg.trainer.evaluation.evaluate_interval = min(cfg.trainer.evaluation.evaluate_interval, 10)
        return cfg

    def invoke(self) -> int:
        """Main training entry point."""
        record_heartbeat()

        assert self.run_dir is not None
        init_file_logging(run_dir=self.run_dir)

        os.makedirs(self.run_dir, exist_ok=True)

        init_logging(run_dir=self.run_dir)

        logger.info(
            f"Training {self.run} on "
            + f"{os.environ.get('NODE_INDEX', '0')}: "
            + f"{os.environ.get('LOCAL_RANK', '0')} ({self.system.device})"
        )

        torch_dist_cfg = setup_torch_distributed(self.system.device)

        logger.info(f"Training {self.run} on {self.system.device}")
        if torch_dist_cfg.is_master:
            with WandbContext(self.wandb, self) as wandb_run:
                handle_train(self, torch_dist_cfg, wandb_run, logger)
        else:
            handle_train(self, torch_dist_cfg, None, logger)

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        return 0


def calculate_default_num_workers(is_serial: bool) -> int:
    """Calculate default number of workers based on hardware."""
    if is_serial:
        return 1

    import multiprocessing

    cpu_count = multiprocessing.cpu_count() or 1

    if torch.cuda.is_available() and torch.distributed.is_initialized():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1

    ideal_workers = (cpu_count // 2) // num_gpus

    # Round down to nearest power of 2
    num_workers = 1
    while num_workers * 2 <= ideal_workers:
        num_workers *= 2

    return max(1, num_workers)


def handle_train(cfg: TrainTool, torch_dist_cfg: TorchDistributedConfig, wandb_run: WandbRun | None, logger: Logger):
    """Handle the training process."""
    assert cfg.run_dir is not None
    assert cfg.run is not None

    # Set default num_workers if not set
    if cfg.trainer.num_workers == 1 and cfg.system.vectorization == "multiprocessing":
        cfg.trainer.num_workers = calculate_default_num_workers(cfg.system.vectorization == "serial")
        logger.info(f"Setting num_workers to {cfg.trainer.num_workers} based on hardware")

    stats_client: StatsClient | None = None
    if cfg.stats_server_uri is not None:
        stats_client = StatsClient.create(cfg.stats_server_uri)

    # Determine git hash for remote simulations
    if cfg.trainer.evaluation and cfg.trainer.evaluation.evaluate_remote:
        if not stats_client:
            cfg.trainer.evaluation.evaluate_remote = False
            logger.info("Not connected to stats server, disabling remote evaluations")
        elif not cfg.trainer.evaluation.evaluate_interval:
            cfg.trainer.evaluation.evaluate_remote = False
            logger.info("Evaluate interval set to 0, disabling remote evaluations")
        elif not cfg.trainer.evaluation.git_hash:
            cfg.trainer.evaluation.git_hash = get_git_hash_for_remote_task(
                skip_git_check=cfg.trainer.evaluation.skip_git_check,
                skip_cmd="trainer.evaluation.skip_git_check=true",
                logger=logger,
            )
            if cfg.trainer.evaluation.git_hash:
                logger.info(f"Git hash for remote evaluations: {cfg.trainer.evaluation.git_hash}")
            else:
                logger.info("No git hash available for remote evaluations")

    # Save configuration
    if os.environ.get("RANK", "0") == "0":  # master only
        with open(os.path.join(cfg.run_dir, "config.json"), "w") as f:
            f.write(cfg.model_dump_json(indent=2))
            logger.info(f"Config saved to {os.path.join(cfg.run_dir, 'config.json')}")

    # Handle distributed training batch scaling
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        if cfg.trainer.scale_batches_by_world_size:
            cfg.trainer.forward_pass_minibatch_target_size = (
                cfg.trainer.forward_pass_minibatch_target_size // world_size
            )
            cfg.trainer.batch_size = cfg.trainer.batch_size // world_size

    policy_store = PolicyStore.create(
        device=cfg.system.device,
        data_dir=cfg.system.data_dir,
        wandb_config=cfg.wandb,
        wandb_run=wandb_run,
    )

    # Use the functional train interface directly
    train(
        run=cfg.run,
        run_dir=cfg.run_dir,
        system_cfg=cfg.system,
        agent_cfg=OmegaConf.create(cfg.policy_architecture),
        device=torch.device(cfg.system.device),
        trainer_cfg=cfg.trainer,
        wandb_run=wandb_run,
        policy_store=policy_store,
        stats_client=stats_client,
        torch_dist_cfg=torch_dist_cfg,
    )
