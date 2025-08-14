#!/usr/bin/env -S uv run

import argparse
import importlib
import logging
import os
import platform
from logging import Logger

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
from metta.util.init.mettagrid_system import init_mettagrid_system_environment

logger = logging.getLogger(__name__)


class TrainToolConfig(Config):
    system: SystemConfig = SystemConfig()
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    wandb: WandbConfig = WandbConfigOff()
    policy_architecture: DictConfig = Field(default_factory=DictConfig)

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

        # Apply Mac overrides if needed
        if not self.bypass_mac_overrides and platform.system() == "Darwin":
            self.apply_mac_overrides()

        # Set up checkpoint and replay directories
        if not self.trainer.checkpoint.checkpoint_dir:
            self.trainer.checkpoint.checkpoint_dir = f"{self.run_dir}/checkpoints/"
        if not self.trainer.simulation.replay_dir:
            self.trainer.simulation.replay_dir = f"{self.run_dir}/replays/"

    def apply_mac_overrides(self):
        """Apply Mac-specific overrides for performance."""
        # Apply system overrides
        self.system.device = "cpu"
        self.system.vectorization = "serial"

        # Apply trainer overrides with minimum values
        self.trainer.minibatch_size = min(self.trainer.minibatch_size, 1024)
        self.trainer.batch_size = min(self.trainer.batch_size, 1024)
        self.trainer.forward_pass_minibatch_target_size = min(self.trainer.forward_pass_minibatch_target_size, 2)
        self.trainer.checkpoint.checkpoint_interval = min(self.trainer.checkpoint.checkpoint_interval, 10)
        self.trainer.checkpoint.wandb_checkpoint_interval = min(self.trainer.checkpoint.wandb_checkpoint_interval, 10)
        self.trainer.bptt_horizon = min(self.trainer.bptt_horizon, 8)
        self.trainer.simulation.evaluate_interval = min(self.trainer.simulation.evaluate_interval, 10)


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
    if cfg.trainer.simulation and cfg.trainer.simulation.evaluate_remote:
        if not stats_client:
            cfg.trainer.simulation.evaluate_remote = False
            logger.info("Not connected to stats server, disabling remote evaluations")
        elif not cfg.trainer.simulation.evaluate_interval:
            cfg.trainer.simulation.evaluate_remote = False
            logger.info("Evaluate interval set to 0, disabling remote evaluations")
        elif not cfg.trainer.simulation.git_hash:
            cfg.trainer.simulation.git_hash = get_git_hash_for_remote_task(
                skip_git_check=cfg.trainer.simulation.skip_git_check,
                skip_cmd="trainer.simulation.skip_git_check=true",
                logger=logger,
            )
            if cfg.trainer.simulation.git_hash:
                logger.info(f"Git hash for remote evaluations: {cfg.trainer.simulation.git_hash}")
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

    # Use the functional train interface directly with failure handling
    try:
        train(
            run=cfg.run,
            run_dir=cfg.run_dir,
            system_cfg=cfg.system,
            agent_cfg=OmegaConf.create(cfg.agent),
            device=torch.device(cfg.system.device),
            trainer_cfg=cfg.trainer,
            wandb_run=wandb_run,
            policy_store=policy_store,
            sim_suite_config=cfg.evals,
            stats_client=stats_client,
        )
    except Exception as training_error:
        # Training failed - check if we have stats info to update status
        # If train() threw an exception before completing, we need to find the run ID some other way
        # For now, this provides the framework for proper failure handling
        logger.error(f"Training failed with error: {training_error}", exc_info=True)

        # Attempt to find and update the most recent training run created by this process
        if stats_client:
            try:
                # This would need implementation to query for the most recent run
                # For demonstration, we log that we would update it
                logger.info("Would attempt to find and update most recent training run status to 'failed'")
            except Exception as e:
                logger.warning(f"Could not update training run status after failure: {e}")

        raise  # Re-raise the original exception


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

    # Initialize the full mettagrid environment
    # Convert to DictConfig for init_mettagrid_system_environment
    init_cfg = DictConfig(
        {
            "seed": cfg.seed,
            "torch_deterministic": cfg.system.torch_deterministic,
            "device": cfg.system.device,
            "vectorization": cfg.system.vectorization,
            "run_dir": cfg.run_dir,
            "dist_cfg_path": None,  # Not using distributed config
        }
    )
    init_mettagrid_system_environment(init_cfg)

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
