#!/usr/bin/env -S uv run

import argparse
import copy
import importlib
import logging
import os
from logging import Logger
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from pydantic import Field
from torch.distributed.elastic.multiprocessing.errors import record

from metta.agent.policy_store import PolicyStore
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.util.config import Config
from metta.common.util.git import get_git_hash_for_remote_task
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.logging_helpers import init_logging
from metta.common.util.stats_client_cfg import get_stats_client
from metta.common.wandb.wandb_context import WandbConfig, WandbConfigOff, WandbContext, WandbRun
from metta.core.distributed import setup_device_and_distributed
from metta.rl.system_config import SystemConfig
from metta.rl.trainer import train
from metta.rl.trainer_config import TrainerConfig

logger = logging.getLogger(__name__)


class TrainToolConfig(Config):
    system: SystemConfig = SystemConfig()
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    wandb: WandbConfig = WandbConfigOff()
    policy_architecture: Any
    run: str
    run_dir: str = Field(default="./train_dir")
    data_dir: str = Field(default="./train_dir")

    # Stats server configuration
    stats_server_uri: str = "https://api.observatory.softmax-research.net"

    # Policy configuration
    policy_uri: str | None = None

    # Optional configurations
    map_preview_uri: str | None = None
    bypass_mac_overrides: bool = False

    # Seed for reproducibility
    seed: int = Field(default=0)

    def model_post_init(self, __context):
        """Post-initialization setup."""
        # Set run_dir based on run name if not explicitly set
        if self.run_dir == "./train_dir":
            self.run_dir = f"{self.data_dir}/{self.run}"

        # Set policy_uri if not set
        if not self.policy_uri:
            self.policy_uri = f"file://{self.run_dir}/checkpoints"

        # Set up checkpoint and replay directories
        if not self.trainer.checkpoint.checkpoint_dir:
            self.trainer.checkpoint.checkpoint_dir = f"{self.run_dir}/checkpoints/"
        if not self.trainer.evaluation.replay_dir:
            self.trainer.evaluation.replay_dir = f"{self.run_dir}/replays/"

    def to_mini(self) -> "TrainToolConfig":
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


def get_policy_store(cfg: TrainToolConfig, wandb_run: WandbRun | None = None) -> PolicyStore:
    """Create policy store from configuration."""
    wandb_config = cfg.wandb

    # Extract entity and project from wandb config if it's enabled
    wandb_entity = None
    wandb_project = None
    if isinstance(wandb_config, WandbConfigOff):
        wandb_entity = None
        wandb_project = None
    else:
        # It's WandbConfigOn which has entity and project
        wandb_entity = getattr(wandb_config, "entity", None)
        wandb_project = getattr(wandb_config, "project", None)

    policy_store = PolicyStore(
        device=cfg.system.device,
        wandb_run=wandb_run,
        data_dir=cfg.data_dir,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        pytorch_cfg=None,  # Not using pytorch config from old system
    )
    return policy_store


def handle_train(cfg: TrainToolConfig, wandb_run: WandbRun | None, logger: Logger):
    """Handle the training process."""

    # Set default num_workers if not set
    if cfg.trainer.num_workers == 1 and cfg.system.vectorization == "multiprocessing":
        cfg.trainer.num_workers = calculate_default_num_workers(cfg.system.vectorization == "serial")

    # Create stats client if configured
    # Create a temporary DictConfig for stats client creation
    stats_cfg = DictConfig(
        {
            "stats_server_uri": cfg.stats_server_uri,
            "run_dir": cfg.run_dir,
        }
    )
    stats_client: StatsClient | None = get_stats_client(stats_cfg, logger)
    if stats_client is not None:
        stats_client.validate_authenticated()

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
        logger.info("Trainer config:\n%s", cfg.trainer.model_dump_json(indent=2))
        os.makedirs(cfg.run_dir, exist_ok=True)
        with open(os.path.join(cfg.run_dir, "config.json"), "w") as f:
            f.write(cfg.model_dump_json(indent=2))

    # Handle distributed training batch scaling
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        if cfg.trainer.scale_batches_by_world_size:
            cfg.trainer.forward_pass_minibatch_target_size = (
                cfg.trainer.forward_pass_minibatch_target_size // world_size
            )
            cfg.trainer.batch_size = cfg.trainer.batch_size // world_size

    policy_store = get_policy_store(cfg, wandb_run)

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
    )


@record
def main(cfg: TrainToolConfig) -> int:
    """Main training entry point."""
    record_heartbeat()

    # Initialize logging
    init_logging(run_dir=cfg.run_dir)

    logger.info(
        f"Training {cfg.run} on "
        + f"{os.environ.get('NODE_INDEX', '0')}: "
        + f"{os.environ.get('LOCAL_RANK', '0')} ({cfg.system.device})"
    )

    # Use shared distributed setup function
    device, is_master, world_size, rank = setup_device_and_distributed(cfg.system.device)

    # Update device to include the local rank if distributed
    cfg.system.device = str(device)

    logger.info(f"Training {cfg.run} on {cfg.system.device}")
    if is_master:
        with WandbContext(cfg.wandb, cfg) as wandb_run:
            handle_train(cfg, wandb_run, logger)
    else:
        handle_train(cfg, None, logger)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=False)
    args = parser.parse_args()

    init_logging()

    if args.cfg:
        with open(args.cfg, "r") as f:
            cfg = TrainToolConfig.model_validate_json(f.read())
    else:
        module_name, func_name = args.func.rsplit(".", 1)
        cfg = importlib.import_module(module_name).__getattribute__(func_name)()

    main(cfg)
