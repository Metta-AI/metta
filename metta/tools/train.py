import logging
import os
import platform
import uuid
from logging import Logger
from typing import Optional

import torch

from metta.agent.agent_config import AgentConfig
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.config.tool import Tool
from metta.common.util.git import get_git_hash_for_remote_task
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.logging_helpers import init_file_logging, init_logging
from metta.common.wandb.wandb_context import WandbConfig, WandbContext, WandbRun
from metta.core.distributed import TorchDistributedConfig, setup_torch_distributed
from metta.rl.trainer import train
from metta.rl.trainer_config import TrainerConfig
from metta.tools.utils.auto_config import auto_replay_dir, auto_stats_server_uri, auto_wandb_config

logger = logging.getLogger(__name__)


def log_on_master(*args, **argv):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(*args, **argv)


class TrainTool(Tool):
    trainer: TrainerConfig = TrainerConfig()
    wandb: WandbConfig = WandbConfig.Unconfigured()
    policy_architecture: Optional[AgentConfig] = None
    run: Optional[str] = None
    run_dir: Optional[str] = None
    stats_server_uri: Optional[str] = auto_stats_server_uri()

    # Policy configuration
    policy_uri: Optional[str] = None

    # Optional configurations
    map_preview_uri: str | None = None
    disable_macbook_optimize: bool = False

    consumed_args: list[str] = ["run"]

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        # Handle run_id being passed via cmd line
        if "run" in args:
            assert self.run is None, "run cannot be set via args and config"
            self.run = args["run"]

        if self.run is None:
            self.run = f"local.{os.getenv('USER', 'unknown')}.{str(uuid.uuid4())}"

        # Set run_dir based on run name if not explicitly set
        if self.run_dir is None:
            self.run_dir = f"{self.system.data_dir}/{self.run}"

        # Set policy_uri if not set
        if not self.policy_uri:
            self.policy_uri = f"file://{self.run_dir}/checkpoints"

        # Set up checkpoint and replay directories
        if not self.trainer.checkpoint.checkpoint_dir:
            self.trainer.checkpoint.checkpoint_dir = f"{self.run_dir}/checkpoints/"

        # Initialize policy_architecture if not provided
        if self.policy_architecture is None:
            self.policy_architecture = AgentConfig()

        if self.wandb == WandbConfig.Unconfigured():
            self.wandb = auto_wandb_config(self.run)

        os.makedirs(self.run_dir, exist_ok=True)

        record_heartbeat()

        init_file_logging(run_dir=self.run_dir)

        init_logging(run_dir=self.run_dir)

        torch_dist_cfg = setup_torch_distributed(self.system.device)

        if not self.trainer.checkpoint.checkpoint_dir:
            self.trainer.checkpoint.checkpoint_dir = f"{self.run_dir}/checkpoints/"

        log_on_master(
            f"Training {self.run} on "
            + f"{os.environ.get('NODE_INDEX', '0')}: "
            + f"{os.environ.get('LOCAL_RANK', '0')} ({self.system.device})"
        )

        log_on_master(f"Training {self.run} on {self.system.device}")
        if torch_dist_cfg.is_master:
            with WandbContext(self.wandb, self) as wandb_run:
                handle_train(self, torch_dist_cfg, wandb_run, logger)
        else:
            handle_train(self, torch_dist_cfg, None, logger)

        # ?? should be in finally block?
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        return 0


def handle_train(cfg: TrainTool, torch_dist_cfg: TorchDistributedConfig, wandb_run: WandbRun | None, logger: Logger):
    assert cfg.run_dir is not None
    assert cfg.run is not None
    run_dir = cfg.run_dir

    _configure_vecenv_settings(cfg)

    stats_client = _configure_evaluation_settings(cfg)

    # Handle distributed training batch scaling
    if torch_dist_cfg.distributed:
        if cfg.trainer.scale_batches_by_world_size:
            cfg.trainer.forward_pass_minibatch_target_size = (
                cfg.trainer.forward_pass_minibatch_target_size // torch_dist_cfg.world_size
            )
            cfg.trainer.batch_size = cfg.trainer.batch_size // torch_dist_cfg.world_size

    if platform.system() == "Darwin":
        cfg = _minimize_config_for_debugging(cfg)

    # Save configuration
    if torch_dist_cfg.is_master:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            f.write(cfg.model_dump_json(indent=2))
            log_on_master(f"Config saved to {os.path.join(run_dir, 'config.json')}")

    # Use the functional train interface directly
    train(
        run=cfg.run,
        run_dir=run_dir,
        system_cfg=cfg.system,
        agent_cfg=cfg.policy_architecture,
        trainer_cfg=cfg.trainer,
        wandb_run=wandb_run,
        stats_client=stats_client,
        torch_dist_cfg=torch_dist_cfg,
    )


def _configure_vecenv_settings(cfg: TrainTool) -> None:
    """Calculate default number of workers based on hardware."""
    if cfg.system.vectorization == "serial":
        cfg.trainer.rollout_workers = 1
        cfg.trainer.async_factor = 1
        return

    ideal_workers = (os.cpu_count() // 2) // torch.cuda.device_count()
    cfg.trainer.rollout_workers = max(1, ideal_workers)


def _configure_evaluation_settings(cfg: TrainTool) -> StatsClient | None:
    if cfg.trainer.evaluation is None:
        return None

    if cfg.trainer.evaluation.replay_dir is None:
        cfg.trainer.evaluation.replay_dir = auto_replay_dir()
        log_on_master(f"Setting replay_dir to {cfg.trainer.evaluation.replay_dir}")

    stats_client: StatsClient | None = None
    if cfg.stats_server_uri is not None:
        stats_client = StatsClient.create(cfg.stats_server_uri)

    # Determine git hash for remote simulations
    if cfg.trainer.evaluation.evaluate_remote:
        if not stats_client:
            cfg.trainer.evaluation.evaluate_remote = False
            log_on_master("Not connected to stats server, disabling remote evaluations")
        elif not cfg.trainer.evaluation.evaluate_interval:
            cfg.trainer.evaluation.evaluate_remote = False
            log_on_master("Evaluate interval set to 0, disabling remote evaluations")
        elif not cfg.trainer.evaluation.git_hash:
            cfg.trainer.evaluation.git_hash = get_git_hash_for_remote_task(
                skip_git_check=cfg.trainer.evaluation.skip_git_check,
                skip_cmd="trainer.evaluation.skip_git_check=true",
                logger=logger,
            )
            if cfg.trainer.evaluation.git_hash:
                log_on_master(f"Git hash for remote evaluations: {cfg.trainer.evaluation.git_hash}")
            else:
                log_on_master("No git hash available for remote evaluations")

    return stats_client


def _minimize_config_for_debugging(cfg: TrainTool) -> TrainTool:
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
