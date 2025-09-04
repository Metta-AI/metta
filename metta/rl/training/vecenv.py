"""Vectorized environment creation for training."""

from typing import Any, Tuple

from metta.cogworks.curriculum import Curriculum
from metta.rl.system_config import SystemConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.vecenv import make_vecenv
from metta.utils.batch import calculate_batch_sizes


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
