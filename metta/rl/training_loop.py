"""Training loop helper functions."""

import logging
from typing import Any, Dict, Tuple

import torch

from metta.common.profiling.stopwatch import Stopwatch
from metta.rich_progress import log_rich_progress, should_use_rich_console
from metta.rl.experience import Experience
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.ppo import ppo
from metta.rl.rollout import rollout
from metta.rl.trainer_config import TrainerConfig

logger = logging.getLogger(__name__)


def should_run(
    epoch: int,
    interval: int,
    is_master: bool = True,
    force: bool = False,
) -> bool:
    """Check if a periodic task should run based on interval and master status."""
    if not is_master or not interval:
        return False

    if force:
        return True

    return epoch % interval == 0


def run_training_epoch(
    vecenv: Any,
    policy: Any,
    optimizer: torch.optim.Optimizer,
    experience: Experience,
    kickstarter: Kickstarter,
    losses: Losses,
    trainer_cfg: TrainerConfig,
    agent_step: int,
    epoch: int,
    device: torch.device,
    timer: Stopwatch,
    world_size: int = 1,
) -> Tuple[int, int, Dict[str, list]]:
    """Run one training epoch (rollout + training).

    Args:
        vecenv: Vectorized environment
        policy: Policy model
        optimizer: Optimizer
        experience: Experience buffer
        kickstarter: Kickstarter for knowledge distillation
        losses: Losses tracker
        trainer_cfg: Training configuration
        agent_step: Current agent step
        epoch: Current epoch
        device: Training device
        timer: Stopwatch timer
        world_size: Number of distributed workers

    Returns:
        Tuple of (new_agent_step, epochs_trained, raw_infos)
    """
    # Rollout phase
    with timer("_rollout"):
        num_steps, raw_infos = rollout(
            vecenv=vecenv,
            policy=policy,
            experience=experience,
            device=device,
            timer=timer,
        )
        agent_step += num_steps * world_size

    # Training phase
    with timer("_train"):
        epochs_trained = ppo(
            policy=policy,
            optimizer=optimizer,
            experience=experience,
            kickstarter=kickstarter,
            losses=losses,
            trainer_cfg=trainer_cfg,
            agent_step=agent_step,
            epoch=epoch,
            device=device,
        )

    return agent_step, epochs_trained, raw_infos


def log_training_progress(
    epoch: int,
    agent_step: int,
    total_timesteps: int,
    steps_per_sec: float,
    train_time: float,
    rollout_time: float,
    stats_time: float,
    is_master: bool,
) -> None:
    """Log training progress with timing breakdown.

    Args:
        epoch: Current epoch
        agent_step: Current agent step
        total_timesteps: Total timesteps to train
        steps_per_sec: Steps per second
        train_time: Time spent training
        rollout_time: Time spent in rollout
        stats_time: Time spent processing stats
        is_master: Whether this is the master rank
    """
    if not is_master:
        return

    total_time = train_time + rollout_time + stats_time
    train_pct = (train_time / total_time) * 100 if total_time > 0 else 0
    rollout_pct = (rollout_time / total_time) * 100 if total_time > 0 else 0
    stats_pct = (stats_time / total_time) * 100 if total_time > 0 else 0

    # Use rich console if appropriate
    if should_use_rich_console():
        log_rich_progress(
            epoch=epoch,
            agent_step=agent_step,
            total_timesteps=total_timesteps,
            steps_per_sec=steps_per_sec,
            train_pct=train_pct,
            rollout_pct=rollout_pct,
            stats_pct=stats_pct,
        )
    else:
        # Format total timesteps for readability
        if total_timesteps >= 1e9:
            total_steps_str = f"{total_timesteps:.0e}"
        else:
            total_steps_str = f"{total_timesteps:,}"

        logger.info(
            f"Epoch {epoch}- "
            f"{steps_per_sec:.0f} SPS- "
            f"step {agent_step}/{total_steps_str}- "
            f"({train_pct:.0f}% train- {rollout_pct:.0f}% rollout- {stats_pct:.0f}% stats)"
        )


def get_epoch_timing(timer: Stopwatch) -> Tuple[float, float, float]:
    """Get timing breakdown for the last epoch.

    Args:
        timer: Stopwatch timer

    Returns:
        Tuple of (rollout_time, train_time, stats_time)
    """
    rollout_time = timer.get_last_elapsed("_rollout")
    train_time = timer.get_last_elapsed("_train")
    stats_time = timer.get_last_elapsed("_process_stats")

    return rollout_time, train_time, stats_time
