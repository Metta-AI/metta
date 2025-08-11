#!/usr/bin/env -S uv run

"""Training script for Metta AI - Refactored example with clean separation."""

import argparse
import logging
import multiprocessing
import platform
import sys
from contextlib import nullcontext
from typing import Optional

import torch
from pydantic import Field
from torch.distributed.elastic.multiprocessing.errors import record

from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.stats_client_cfg import get_stats_client
from metta.common.util.typed_config import ConfigWithBuilder
from metta.common.wandb.wandb_config import WandbConfig
from metta.common.wandb.wandb_context import WandbContext
from metta.core.distributed import setup_device_and_distributed
from metta.rl.env_config import create_env_config
from metta.rl.trainer import train
from metta.rl.trainer_config import TrainerConfig
from metta.sim.simulation_config import SimulationSuiteConfig
from tools.utils import get_policy_store_from_cfg

logger = logging.getLogger(__name__)


class TrainToolConfig(ConfigWithBuilder):
    """Configuration for the training tool - ONLY contains other configs."""

    # Config objects only - no runtime data
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    evals: Optional[SimulationSuiteConfig] = Field(default=None, description="Evaluation suite config")
    wandb: WandbConfig = Field(default_factory=WandbConfig, description="WandB configuration")


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


def apply_mac_device_overrides(device: str, trainer_cfg: TrainerConfig, bypass: bool = False) -> str:
    """Apply Mac-specific device overrides.

    Args:
        device: Current device setting
        trainer_cfg: Trainer configuration to modify
        bypass: Whether to bypass Mac overrides

    Returns:
        Updated device string
    """
    if not bypass and platform.system() == "Darwin":
        device = "cpu"
        # Apply serial vectorization settings (async_factor=1, zero_copy=False)
        trainer_cfg.async_factor = 1
        trainer_cfg.zero_copy = False
    return device


@record
def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train a Metta AI agent")

    # Configuration file (contains only Config objects)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")

    # Runtime parameters (not stored in config file)
    parser.add_argument("--run", type=str, required=True, help="Run name/identifier")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--bypass-mac-overrides", action="store_true", help="Bypass Mac device overrides")
    parser.add_argument("--map-preview-uri", type=str, default=None, help="Map preview URI")

    # Runtime overrides for specific config fields
    parser.add_argument("--num-workers", type=int, default=None, help="Override number of workers")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Override checkpoint interval")

    args = parser.parse_args()

    # Load configuration from file (only Config objects)
    cfg = TrainToolConfig.from_file(args.config)

    # Apply runtime overrides to config objects
    if args.num_workers:
        cfg.trainer.num_workers = args.num_workers

    if args.total_timesteps:
        cfg.trainer.total_timesteps = args.total_timesteps

    if args.checkpoint_interval and cfg.trainer.checkpoint:
        cfg.trainer.checkpoint.checkpoint_interval = args.checkpoint_interval

    # Apply Mac device overrides (modifies device and trainer config)
    device = apply_mac_device_overrides(args.device, cfg.trainer, args.bypass_mac_overrides)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info(f"Starting training with run: {args.run}")
    logger.info(f"Device: {device}")
    logger.info(f"Config loaded from: {args.config}")

    # Setup distributed training
    setup_device_and_distributed(device)

    # Calculate default workers if not set
    if cfg.trainer.num_workers is None:
        cfg.trainer.num_workers = _calculate_default_num_workers(
            cfg.trainer.async_factor == 1 and not cfg.trainer.zero_copy
        )
        logger.info(f"Auto-detected num_workers: {cfg.trainer.num_workers}")

    # Setup WandB context
    wandb_ctx = WandbContext.from_config(cfg.wandb, args.run) if cfg.wandb.enabled else nullcontext()

    with wandb_ctx as wandb_run:
        # Get policy store
        policy_store = get_policy_store_from_cfg(
            {
                "device": device,
                "wandb": cfg.wandb.model_dump() if cfg.wandb else {},
                "map_preview_uri": args.map_preview_uri,
            }
        )

        # Create environment config
        env_config = create_env_config(device=device)

        # Get stats client
        stats_client = get_stats_client(cfg.trainer.stats_client)

        # Train the model
        train(
            run_name=args.run,
            trainer_config=cfg.trainer,
            env_config=env_config,
            policy_store=policy_store,
            eval_suite_config=cfg.evals,
            stats_client=stats_client,
            wandb_run=wandb_run,
        )

        # Record heartbeat for monitoring
        record_heartbeat(args.run, "training_completed")

    logger.info("Training completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
