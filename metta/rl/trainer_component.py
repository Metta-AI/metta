"""Component-based trainer interface that replaces the functional train interface."""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import DictConfig

from metta.agent.policy_store import PolicyStore
from metta.app_backend.stats_client import StatsClient
from metta.common.wandb.wandb_context import WandbRun
from metta.rl.components import Trainer
from metta.rl.trainer_config import create_trainer_config
from metta.sim.simulation_config import SimulationSuiteConfig

logger = logging.getLogger(__name__)


def train(
    cfg: Any,
    wandb_run: Optional[WandbRun],
    policy_store: PolicyStore,
    sim_suite_config: SimulationSuiteConfig,
    stats_client: Optional[StatsClient] = None,
) -> None:
    """Train using component-based architecture.

    This function provides backward compatibility with the existing functional interface
    while using the new component-based Trainer internally.

    Args:
        cfg: Configuration object
        wandb_run: Optional wandb run
        policy_store: Policy store (will be replaced by trainer's internal one)
        sim_suite_config: Simulation suite configuration
        stats_client: Optional stats client (not yet integrated)
    """
    logger.info(f"run_dir = {cfg.run_dir}")

    # Log recent checkpoints
    checkpoints_dir = Path(cfg.run_dir) / "checkpoints"
    if checkpoints_dir.exists():
        files = sorted(os.listdir(checkpoints_dir))
        recent_files = files[-3:] if len(files) >= 3 else files
        logger.info(f"Recent checkpoints: {', '.join(recent_files)}")

    # Create trainer config
    trainer_cfg = create_trainer_config(cfg)

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

    # Create trainer with component architecture
    trainer = Trainer(
        trainer_config=trainer_cfg,
        run_dir=str(run_dir),
        run_name=cfg.run,
        checkpoint_dir=str(checkpoint_dir),
        replay_dir=str(replay_dir),
        stats_dir=str(stats_dir),
        device=device,
        wandb_config=cfg.wandb if is_master and isinstance(cfg.wandb, DictConfig) else None,
        global_config=cfg if is_master else None,
    )

    # Note: The trainer creates its own policy store internally, so the passed one is not used
    # This maintains backward compatibility but the passed policy_store is ignored

    if stats_client is not None:
        logger.warning("Stats client not yet integrated with component-based trainer")

    try:
        # Set up trainer components
        vectorization = cfg.get("vectorization", "multiprocessing")
        trainer.setup(vectorization=vectorization)

        # Run training
        trainer.train()

    finally:
        # Clean up
        trainer.cleanup()
