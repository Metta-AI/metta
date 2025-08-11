#!/usr/bin/env -S uv run

"""Training script for Metta AI using typer and Pydantic configs."""

import logging
import multiprocessing
import platform
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import typer
from pydantic import Field
from torch.distributed.elastic.multiprocessing.errors import record

from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.stats_client_cfg import get_stats_client
from metta.common.util.typed_config import ConfigWithBuilder
from metta.common.wandb.wandb_config import WandbConfig
from metta.common.wandb.wandb_context import WandbContext, WandbRun
from metta.core.distributed import setup_device_and_distributed
from metta.rl.env_config import create_env_config
from metta.rl.trainer import train
from metta.rl.trainer_config import TrainerConfig
from metta.sim.simulation_config import SimulationSuiteConfig
from tools.utils import get_policy_store_from_cfg

logger = logging.getLogger(__name__)
app = typer.Typer()


class TrainConfig(ConfigWithBuilder):
    """Configuration for the training script."""

    # Run configuration
    run: str = Field(description="Run name/identifier")

    # Training configuration - using TrainerConfig as a nested field
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)

    # Evaluation configuration
    evals: SimulationSuiteConfig | None = Field(default=None, description="Evaluation suite config")

    # Other configurations
    wandb: WandbConfig = Field(default_factory=WandbConfig, description="WandB configuration")
    device: str = Field(default="cuda", description="Device to use (cuda/cpu)")
    bypass_mac_overrides: bool = Field(default=False, description="Bypass Mac device overrides")

    # Paths
    map_preview_uri: str | None = Field(default=None, description="Map preview URI")


def _calculate_default_num_workers(is_serial: bool) -> int:
    """Calculate default number of workers based on hardware."""
    if is_serial:
        return 1

    cpu_count = multiprocessing.cpu_count() or 1

    if torch.cuda.is_available() and torch.distributed.is_initialized():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1 if torch.cuda.is_available() else 0

    # Prefer 2 workers per GPU, fallback to CPU count
    if num_gpus > 0:
        return num_gpus * 2
    else:
        return cpu_count


def apply_mac_device_overrides(cfg: TrainConfig) -> None:
    """Apply Mac-specific device overrides."""
    if not cfg.bypass_mac_overrides and platform.system() == "Darwin":
        cfg.device = "cpu"
        # Apply serial vectorization settings (async_factor=1, zero_copy=False)
        cfg.trainer.async_factor = 1
        cfg.trainer.zero_copy = False


@app.command()
@record
def main(
    config: str = typer.Option(..., "--config", help="Path to YAML configuration file"),
    device: Optional[str] = typer.Option(None, "--device", help="Override device (cuda/cpu)"),
    num_workers: Optional[int] = typer.Option(None, "--num-workers", help="Override number of workers"),
):
    """Train a Metta AI agent."""
    # Load configuration from file
    cfg = TrainConfig.from_file(config)

    # Apply command line overrides
    if device:
        cfg.device = device

    if num_workers:
        cfg.trainer.num_workers = num_workers

    # Apply Mac overrides
    apply_mac_device_overrides(cfg)

    # Set default num_workers if not specified
    if not hasattr(cfg.trainer, "num_workers") or cfg.trainer.num_workers <= 0:
        # Check if we're in serial mode (async_factor=1 and zero_copy=False indicates serial)
        is_serial = cfg.trainer.async_factor == 1 and not cfg.trainer.zero_copy
        cfg.trainer.num_workers = _calculate_default_num_workers(is_serial)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info(f"Starting training with config: {cfg.run}")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"Num workers: {cfg.trainer.num_workers}")
    logger.info(f"Total timesteps: {cfg.trainer.total_timesteps}")

    # Setup device and distributed training
    setup_device_and_distributed(cfg.device)

    # Set checkpoint and replay directories based on run name
    train_dir = Path("train_dir") / cfg.run
    train_dir.mkdir(parents=True, exist_ok=True)

    cfg.trainer.checkpoint.checkpoint_dir = str(train_dir / "checkpoints")
    cfg.trainer.simulation.replay_dir = str(train_dir / "replays")

    # Create checkpoint directory
    Path(cfg.trainer.checkpoint.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.trainer.simulation.replay_dir).mkdir(parents=True, exist_ok=True)

    # Create env config
    if cfg.trainer.env:
        env_config = create_env_config(cfg.trainer.env) if isinstance(cfg.trainer.env, str) else cfg.trainer.env
    else:
        # Create default env config if none specified
        from metta.rl.env_config import EnvConfig

        env_config = EnvConfig(device=cfg.device)

    # Setup WandB context
    wandb_enabled = cfg.wandb.enabled
    wandb_context = None
    wandb_run = None

    if wandb_enabled:
        wandb_run = WandbRun(
            project=cfg.wandb.project,
            name=cfg.wandb.name or cfg.run,
            config=cfg.model_dump(),
            tags=cfg.wandb.tags,
            entity=cfg.wandb.entity,
            run_id=cfg.wandb.run_id,
        )
        wandb_context = WandbContext(wandb_run)

    # Get policy store - create a config object with the fields the function expects
    from types import SimpleNamespace

    policy_config = SimpleNamespace()
    policy_config.device = cfg.device
    policy_config.wandb = cfg.wandb
    policy_config.data_dir = getattr(cfg.trainer, "data_dir", None)

    policy_store = get_policy_store_from_cfg(policy_config, wandb_run)

    # Setup stats client
    from omegaconf import OmegaConf

    trainer_dict = OmegaConf.create(cfg.trainer.model_dump())
    stats_client = get_stats_client(trainer_dict, logger)

    # Record heartbeat
    record_heartbeat()

    try:
        # Start training
        with wandb_context if wandb_context else nullcontext():
            # Create run directory
            run_dir = f"./train_dir/{cfg.run}"
            Path(run_dir).mkdir(parents=True, exist_ok=True)

            # Create agent config (placeholder for now)
            from omegaconf import OmegaConf

            agent_cfg = OmegaConf.create({})

            train(
                run_dir=run_dir,
                run=cfg.run,
                env_cfg=env_config,
                agent_cfg=agent_cfg,
                device=torch.device(cfg.device),
                trainer_cfg=cfg.trainer,
                wandb_run=wandb_run,
                policy_store=policy_store,
                sim_suite_config=cfg.evals,
                stats_client=stats_client,
            )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    logger.info("Training completed successfully")
    return 0


if __name__ == "__main__":
    app()
