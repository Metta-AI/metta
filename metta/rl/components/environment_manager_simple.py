"""Simplified environment manager that uses interface.Environment."""

import logging
from typing import Any, Optional

import torch

from metta.interface.environment import Environment
from metta.rl.trainer_config import TrainerConfig

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
        self._curriculum = None

    def create_environment(
        self,
        curriculum_path: Optional[str] = None,
        vectorization: str = "multiprocessing",
        is_training: bool = True,
        seed: Optional[int] = None,
        rank: int = 0,
    ) -> Any:
        """Create and configure the environment using interface.Environment.

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

        logger.info(f"Creating environment with curriculum: {curriculum}")

        # Import batch size calculation utility
        from metta.rl.util.batch_utils import calculate_batch_sizes

        # Default num_agents - will be updated from actual environment
        self._num_agents = 4

        # Calculate batch sizes properly like run.py does
        target_batch_size, batch_size, num_envs = calculate_batch_sizes(
            forward_pass_minibatch_target_size=self.trainer_config.forward_pass_minibatch_target_size,
            num_agents=self._num_agents,
            num_workers=self.num_workers,
            async_factor=self.trainer_config.async_factor,
        )

        # Use calculated values but ensure minimum for Experience buffer
        # Experience requires batch_size >= num_agents * bptt_horizon
        min_batch_size = self._num_agents * self.trainer_config.bptt_horizon
        if batch_size < min_batch_size:
            logger.info(f"Calculated batch_size {batch_size} too small, using minimum {min_batch_size}")
            self._batch_size = min_batch_size
            self._num_envs = self._batch_size // self._num_agents
        else:
            self._batch_size = batch_size
            self._num_envs = num_envs
        actual_batch_size = self._batch_size

        # Create environment using interface.Environment
        logger.info(
            f"Creating Environment with batch_size={actual_batch_size},\
            num_envs={self._num_envs}, vectorization={vectorization}"
        )
        self._env = Environment(
            curriculum_path=curriculum,
            num_agents=self._num_agents,
            width=32,
            height=32,
            device=str(self.device),
            seed=seed,
            num_envs=self._num_envs,
            num_workers=self.num_workers,
            batch_size=actual_batch_size,
            async_factor=self.trainer_config.async_factor,
            zero_copy=self.trainer_config.zero_copy,
            is_training=is_training,
            vectorization=vectorization,
        )

        # Update num_agents from actual environment
        if hasattr(self._env, "driver_env"):
            driver_env = self._env.driver_env
            if hasattr(driver_env, "_curriculum"):
                self._curriculum = driver_env._curriculum
                task = self._curriculum.get_task()
                if hasattr(task, "env_cfg"):
                    actual_num_agents = task.env_cfg().game.num_agents
                    if actual_num_agents != self._num_agents:
                        logger.info(f"Updating num_agents from {self._num_agents} to {actual_num_agents}")
                        self._num_agents = actual_num_agents
                        # Update batch size to match
                        self._batch_size = self._num_envs * self._num_agents
                        logger.info(f"Updated batch_size to {self._batch_size}")

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
        return self.env.driver_env

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
        return self._curriculum or getattr(self.driver_env, "_curriculum", None)

    def close(self) -> None:
        """Close the environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
