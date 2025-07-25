#!/usr/bin/env -S uv run
"""Training script using component-based architecture."""

import logging
import multiprocessing
import os
from logging import Logger
from pathlib import Path

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record

from metta.app_backend.stats_client import StatsClient
from metta.common.util.config import Config
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.stats_client_cfg import get_stats_client
from metta.common.wandb.wandb_context import WandbContext, WandbRun
from metta.rl.components import Trainer
from metta.rl.util.distributed import setup_device_and_distributed
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.util.metta_script import metta_script
from tools.sweep_config_utils import (
    load_train_job_config_with_overrides,
    validate_train_job_config,
)

logger = logging.getLogger(__name__)


# TODO: populate this more
class TrainJob(Config):
    __init__ = Config.__init__
    evals: SimulationSuiteConfig
    map_preview_uri: str | None = None


def _calculate_default_num_workers(is_serial: bool) -> int:
    if is_serial:
        return 1

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


def train(cfg: DictConfig | ListConfig, wandb_run: WandbRun | None, logger: Logger):
    cfg = load_train_job_config_with_overrides(cfg)

    # Validation must be done after merging
    if not cfg.trainer.num_workers:
        cfg.trainer.num_workers = _calculate_default_num_workers(cfg.vectorization == "serial")
    cfg = validate_train_job_config(cfg)

    logger.info("Trainer config after overrides:\n%s", OmegaConf.to_yaml(cfg.trainer, resolve=True))

    if os.environ.get("RANK", "0") == "0":
        with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

    TrainJob(cfg.train_job)

    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        if cfg.trainer.scale_batches_by_world_size:
            cfg.trainer.forward_pass_minibatch_target_size = (
                cfg.trainer.forward_pass_minibatch_target_size // world_size
            )
            cfg.trainer.batch_size = cfg.trainer.batch_size // world_size

    # Extract directories from config
    run_dir = Path(cfg.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    replay_dir = run_dir / "replays"
    stats_dir = run_dir / "stats"

    # Ensure directories exist
    for dir_path in [checkpoint_dir, replay_dir, stats_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Get device from config
    device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device

    # Get is_master
    is_master = int(os.environ.get("RANK", 0)) == 0

    # Get stats client
    stats_client: StatsClient | None = get_stats_client(cfg, logger)
    if stats_client is not None:
        stats_client.validate_authenticated()

    # Create trainer with component architecture
    trainer = Trainer(
        trainer_config=cfg.trainer,
        run_dir=str(run_dir),
        run_name=cfg.run,
        checkpoint_dir=str(checkpoint_dir),
        replay_dir=str(replay_dir),
        stats_dir=str(stats_dir),
        device=device,
        wandb_config=cfg.wandb if is_master else None,
        global_config=cfg if is_master else None,
        stats_client=stats_client,
    )

    try:
        # Set up trainer components with seed from config
        seed = cfg.get("seed", None)
        trainer.setup(vectorization=cfg.vectorization, seed=seed)

        # Run training
        trainer.train()

    finally:
        # Clean up
        trainer.cleanup()


@record
def main(cfg: DictConfig) -> int:
    record_heartbeat()

    logger.info(
        f"Training {cfg.run} on "
        + f"{os.environ.get('NODE_INDEX', '0')}: "
        + f"{os.environ.get('LOCAL_RANK', '0')} ({cfg.device})"
    )

    # Use shared distributed setup function
    device, is_master, world_size, rank = setup_device_and_distributed(cfg.device)

    # Update cfg.device to include the local rank if distributed
    cfg.device = str(device)

    logger.info(f"Training {cfg.run} on {cfg.device}")
    if is_master:
        logger.info(f"Train job config: {OmegaConf.to_yaml(cfg, resolve=True)}")

        # Initialize wandb using WandbContext
        with WandbContext(cfg.wandb, cfg) as wandb_run:
            train(cfg, wandb_run, logger)
    else:
        train(cfg, None, logger)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    return 0


metta_script(main, "train_job")
