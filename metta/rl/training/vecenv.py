"""Vectorized environment creation for training."""

import os
from typing import Any, Tuple

import torch

from metta.cogworks.curriculum import Curriculum
from metta.rl.system_config import SystemConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.vecenv import make_vecenv
from metta.utils.batch import calculate_batch_sizes


def configure_rollout_workers(system_cfg: SystemConfig, trainer_cfg: TrainerConfig) -> None:
    """Calculate and set default number of workers based on hardware.

    This function modifies trainer_cfg in-place to set appropriate
    rollout workers and async factor based on the system configuration
    and available hardware.

    Args:
        system_cfg: System configuration with vectorization settings
        trainer_cfg: Trainer configuration to modify in-place
    """
    if system_cfg.vectorization == "serial":
        trainer_cfg.rollout_workers = 1
        trainer_cfg.async_factor = 1
        return

    num_gpus = torch.cuda.device_count() or 1  # fallback to 1 to avoid division by zero
    cpu_count = os.cpu_count() or 1  # fallback to 1 to avoid division by None
    ideal_workers = (cpu_count // 2) // num_gpus
    trainer_cfg.rollout_workers = max(1, ideal_workers)


def create(
    curriculum: Curriculum,
    system_cfg: SystemConfig,
    trainer_cfg: TrainerConfig,
    rank: int,
) -> Tuple[Any, int, int, int]:
    """Create and setup vectorized environment for training.

    Args:
        curriculum: Curriculum for task selection
        system_cfg: System configuration
        trainer_cfg: Trainer configuration
        rank: Current process rank

    Returns:
        Tuple of (vecenv, target_batch_size, batch_size, num_envs)
    """
    # Get number of agents from the current task
    num_agents = curriculum.get_task().get_env_cfg().game.num_agents

    # Calculate batch sizes
    target_batch_size, batch_size, num_envs = calculate_batch_sizes(
        forward_pass_minibatch_target_size=trainer_cfg.forward_pass_minibatch_target_size,
        num_agents=num_agents,
        num_workers=trainer_cfg.rollout_workers,
        async_factor=trainer_cfg.async_factor,
    )

    # Create vectorized environment
    vecenv = make_vecenv(
        curriculum,
        system_cfg.vectorization,
        num_envs=num_envs,
        batch_size=batch_size,
        num_workers=trainer_cfg.rollout_workers,
        zero_copy=trainer_cfg.zero_copy,
        is_training=True,
    )

    # Initialize environment with seed
    vecenv.async_reset(system_cfg.seed + rank)

    return vecenv, target_batch_size, batch_size, num_envs
