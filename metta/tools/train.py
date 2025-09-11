import os
import platform
from typing import Optional

import torch

import gitta as git
from metta.agent.agent_config import AgentConfig
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.tool import Tool
from metta.common.util.git_repo import REPO_SLUG
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.log_config import getRankAwareLogger, init_logging
from metta.common.wandb.wandb_context import WandbConfig, WandbContext, WandbRun
from metta.core.distributed import TorchDistributedConfig, cleanup_distributed, setup_torch_distributed
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.trainer import train
from metta.rl.trainer_config import TrainerConfig
from metta.tools.utils.auto_config import auto_replay_dir, auto_run_name, auto_stats_server_uri, auto_wandb_config

logger = getRankAwareLogger(__name__)


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

    consumed_args: list[str] = ["run", "group"]

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        init_logging(run_dir=self.run_dir)

        # Handle run_id being passed via cmd line
        if "run" in args:
            assert self.run is None, "run cannot be set via args and config"
            self.run = args["run"]

        if self.run is None:
            self.run = auto_run_name(prefix="local")
        group_override = args.get("group")

        # Set run_dir based on run name if not explicitly set
        if self.run_dir is None:
            self.run_dir = f"{self.system.data_dir}/{self.run}"

        # Set policy_uri if not set
        if not self.policy_uri:
            self.policy_uri = CheckpointManager.normalize_uri(f"{self.run_dir}/checkpoints")

        # Set up checkpoint and replay directories
        if not self.trainer.checkpoint.checkpoint_dir:
            self.trainer.checkpoint.checkpoint_dir = f"{self.run_dir}/checkpoints/"

        # Initialize policy_architecture if not provided
        if self.policy_architecture is None:
            self.policy_architecture = AgentConfig()

        if self.wandb == WandbConfig.Unconfigured():
            self.wandb = auto_wandb_config(self.run)

        # Override group if provided via args (for sweep support)
        if group_override:
            self.wandb.group = group_override

        os.makedirs(self.run_dir, exist_ok=True)

        record_heartbeat()

        torch_dist_cfg = setup_torch_distributed(self.system.device)

        if not self.trainer.checkpoint.checkpoint_dir:
            self.trainer.checkpoint.checkpoint_dir = f"{self.run_dir}/checkpoints/"

        logger.info_master(
            f"Training {self.run} on "
            + f"{os.environ.get('NODE_INDEX', '0')}: "
            + f"{os.environ.get('LOCAL_RANK', '0')} ({self.system.device})",
        )

        logger.info_master(
            f"Training {self.run} on {self.system.device}",
        )
        if torch_dist_cfg.is_master:
            with WandbContext(self.wandb, self) as wandb_run:
                handle_train(self, torch_dist_cfg, wandb_run)
        else:
            handle_train(self, torch_dist_cfg, None)

        cleanup_distributed()

        return 0


def handle_train(cfg: TrainTool, torch_dist_cfg: TorchDistributedConfig, wandb_run: WandbRun | None) -> None:
    assert cfg.run_dir is not None
    assert cfg.run is not None
    run_dir = cfg.run_dir

    _configure_vecenv_settings(cfg)

    stats_client: StatsClient | None = None
    if cfg.stats_server_uri is not None:
        stats_client = StatsClient.create(cfg.stats_server_uri)

    _configure_evaluation_settings(cfg, stats_client)

    # Handle distributed training batch scaling
    if torch_dist_cfg.distributed:
        if cfg.trainer.scale_batches_by_world_size:
            cfg.trainer.forward_pass_minibatch_target_size = (
                cfg.trainer.forward_pass_minibatch_target_size // torch_dist_cfg.world_size
            )
            cfg.trainer.batch_size = cfg.trainer.batch_size // torch_dist_cfg.world_size

    checkpoint_manager = CheckpointManager(run=cfg.run, run_dir=cfg.run_dir)

    if platform.system() == "Darwin" and not cfg.disable_macbook_optimize:
        cfg = _minimize_config_for_debugging(cfg)

    # Save configuration
    if torch_dist_cfg.is_master:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            f.write(cfg.model_dump_json(indent=2))
            logger.info_master(f"Config saved to {os.path.join(run_dir, 'config.json')}")

    assert cfg.run
    assert cfg.policy_architecture

    # Use the functional train interface directly
    train(
        run=cfg.run,
        run_dir=run_dir,
        system_cfg=cfg.system,
        agent_cfg=cfg.policy_architecture,
        device=torch.device(cfg.system.device),
        trainer_cfg=cfg.trainer,
        wandb_run=wandb_run,
        checkpoint_manager=checkpoint_manager,
        stats_client=stats_client,
        torch_dist_cfg=torch_dist_cfg,
    )


def _configure_vecenv_settings(cfg: TrainTool) -> None:
    """Calculate default number of workers based on hardware."""
    if cfg.system.vectorization == "serial":
        cfg.trainer.rollout_workers = 1
        cfg.trainer.async_factor = 1
        return

    num_gpus = torch.cuda.device_count() or 1  # fallback to 1 to avoid division by zero
    cpu_count = os.cpu_count() or 1  # fallback to 1 to avoid division by None
    ideal_workers = (cpu_count // 2) // num_gpus
    cfg.trainer.rollout_workers = max(1, ideal_workers)


def _configure_evaluation_settings(cfg: TrainTool, stats_client: StatsClient | None) -> None:
    if cfg.trainer.evaluation is None:
        return

    if cfg.trainer.evaluation.replay_dir is None:
        cfg.trainer.evaluation.replay_dir = auto_replay_dir()
        logger.info_master(f"Setting replay_dir to {cfg.trainer.evaluation.replay_dir}")

    if cfg.stats_server_uri is not None and stats_client is None:
        stats_client = StatsClient.create(cfg.stats_server_uri)

    # Determine git hash for remote simulations
    if cfg.trainer.evaluation.evaluate_remote:
        if not stats_client:
            cfg.trainer.evaluation.evaluate_remote = False
            logger.info_master("Not connected to stats server, disabling remote evaluations")
        elif not cfg.trainer.evaluation.evaluate_interval:
            cfg.trainer.evaluation.evaluate_remote = False
            logger.info_master("Evaluate interval set to 0, disabling remote evaluations")
        elif not cfg.trainer.evaluation.git_hash:
            cfg.trainer.evaluation.git_hash = git.get_git_hash_for_remote_task(
                target_repo=REPO_SLUG,
                skip_git_check=cfg.trainer.evaluation.skip_git_check,
                skip_cmd="trainer.evaluation.skip_git_check=true",
            )
            if cfg.trainer.evaluation.git_hash:
                logger.info_master(f"Git hash for remote evaluations: {cfg.trainer.evaluation.git_hash}")
            else:
                logger.info_master("No git hash available for remote evaluations")


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
