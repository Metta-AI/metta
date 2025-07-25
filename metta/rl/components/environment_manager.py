"""Manages environment creation and interaction."""

import logging
from typing import Any, Optional

import numpy as np
import torch

from metta.rl.trainer_config import TrainerConfig
from metta.rl.util.batch_utils import calculate_batch_sizes
from metta.rl.vecenv import make_vecenv

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Manages environment creation and configuration."""

    def __init__(
        self,
        trainer_config: TrainerConfig,
        device: torch.device,
        num_workers: Optional[int] = None,
    ):
        """Initialize environment manager.

        Args:
            trainer_config: Training configuration
            device: Device to run computations on
            num_workers: Optional override for number of workers
        """
        self.trainer_config = trainer_config
        self.device = device
        self.num_workers = num_workers or trainer_config.num_workers
        self._env = None
        self._num_agents = None
        self._batch_size = None
        self._num_envs = None

    def create_environment(
        self,
        curriculum_path: Optional[str] = None,
        vectorization: str = "multiprocessing",
        is_training: bool = True,
        seed: Optional[int] = None,
        rank: int = 0,
    ) -> Any:
        """Create and configure the environment.

        Args:
            curriculum_path: Optional override for curriculum path
            vectorization: Vectorization mode ("serial", "multiprocessing", etc.)
            is_training: Whether this is for training (affects batch size calculation)
            seed: Optional seed for environment reset
            rank: Process rank for distributed training

        Returns:
            Configured vecenv instance
        """
        # Use provided curriculum or default from config
        curriculum = curriculum_path or self.trainer_config.curriculum

        # Import here to avoid circular dependency
        from omegaconf import DictConfig

        from metta.mettagrid.curriculum.util import curriculum_from_config_path

        # Create curriculum object
        curr_obj = curriculum_from_config_path(curriculum, DictConfig(self.trainer_config.env_overrides))

        # For training, we need to calculate batch sizes first
        # We need to know num_agents, so let's peek at the curriculum
        if is_training:
            self._num_agents = curr_obj.get_task().env_cfg().game.num_agents

            # Calculate batch sizes
            target_batch_size, batch_size, num_envs = calculate_batch_sizes(
                forward_pass_minibatch_target_size=self.trainer_config.forward_pass_minibatch_target_size,
                num_agents=self._num_agents,
                num_workers=self.num_workers,
                async_factor=self.trainer_config.async_factor,
            )
            self._batch_size = batch_size
            self._num_envs = num_envs
        else:
            # For non-training uses, use defaults
            self._num_agents = 4  # Default
            self._batch_size = 1024
            self._num_envs = 256

        # Create vecenv using make_vecenv like the original
        self._env = make_vecenv(
            curr_obj,
            vectorization,
            num_envs=self._num_envs,
            batch_size=self._batch_size,
            num_workers=self.num_workers,
            zero_copy=self.trainer_config.zero_copy,
            is_training=is_training,
        )

        # Reset with seed like the original
        if seed is None:
            seed = np.random.randint(0, 1000000)
        self._env.async_reset(seed + rank)

        return self._env

    @property
    def env(self) -> Any:
        """Get the current environment, creating if needed."""
        if self._env is None:
            self.create_environment()
        return self._env

    @property
    def driver_env(self) -> Any:
        """Get the underlying driver environment (MettaGridEnv)."""
        return self.env.driver_env  # type: ignore[attr-defined]

    @property
    def num_agents(self) -> int:
        """Get number of agents in the environment."""
        if self._num_agents is None:
            _ = self.env  # Force creation
        return self._num_agents

    @property
    def batch_size(self) -> int:
        """Get the batch size for this environment."""
        if self._batch_size is None:
            _ = self.env  # Force creation
        return self._batch_size

    @property
    def num_envs(self) -> int:
        """Get number of parallel environments."""
        if self._num_envs is None:
            _ = self.env  # Force creation
        return self._num_envs

    def get_curriculum(self) -> Any:
        """Get the curriculum object from the environment."""
        return getattr(self.driver_env, "_curriculum", None)

    def close(self) -> None:
        """Close the environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
