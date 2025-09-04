"""Training environment wrapper for vectorized environments."""

import logging
from typing import Any, Optional, Tuple

import torch
from pydantic import Field

from metta.cogworks.curriculum import Curriculum
from metta.mettagrid.config import Config
from metta.rl.experience import Experience
from metta.rl.training.curriculum_config import CurriculumConfig
from metta.rl.vecenv import make_vecenv
from metta.utils.batch import calculate_batch_sizes

logger = logging.getLogger(__name__)


class TrainingEnvironmentConfig(Config):
    """Configuration for training environment."""

    num_workers: int = Field(default=1, ge=1)
    """Number of parallel workers for environment"""

    async_factor: int = Field(default=1, ge=1)
    """Async factor for environment parallelization"""

    forward_pass_minibatch_target_size: int = Field(default=4096, gt=0)
    """Target size for forward pass minibatches"""

    zero_copy: bool = Field(default=False)
    """Whether to use zero-copy optimization"""

    vectorization: str = Field(default="parallel")
    """Vectorization mode: 'serial' or 'parallel'"""

    seed: int = Field(default=0)
    """Random seed for environment"""

    curriculum: CurriculumConfig = Field(default=None)
    """Curriculum configuration for task selection"""


class TrainingEnvironment:
    """Manages the vectorized training environment and experience generation."""

    def __init__(
        self,
        config: TrainingEnvironmentConfig,
        rank: int = 0,
    ):
        """Initialize training environment.

        Args:
            config: Training environment configuration
            curriculum: Curriculum for task selection (uses config.curriculum if not provided)
            rank: Process rank for distributed training
        """
        self.config = config

        self.curriculum = Curriculum(config.curriculum)
        self.rank = rank

        # Initialize vecenv variables
        self.vecenv: Optional[Any] = None
        self.metta_grid_env = None
        self.target_batch_size = None
        self.batch_size = None
        self.num_envs = None
        self.num_agents = None

    def setup(self) -> Tuple[Any, int, int, int]:
        """Create and setup vectorized environment.

        Returns:
            Tuple of (metta_grid_env, target_batch_size, batch_size, num_envs)
        """
        # Get number of agents from the current task
        self.num_agents = self.curriculum.get_task().get_env_cfg().game.num_agents

        # Calculate batch sizes
        self.target_batch_size, self.batch_size, self.num_envs = calculate_batch_sizes(
            forward_pass_minibatch_target_size=self.config.forward_pass_minibatch_target_size,
            num_agents=self.num_agents,
            num_workers=self.config.num_workers,
            async_factor=self.config.async_factor,
        )

        # Create vectorized environment
        self.vecenv = make_vecenv(
            self.curriculum,
            self.config.vectorization,
            num_envs=self.num_envs,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            zero_copy=self.config.zero_copy,
            is_training=True,
        )

        # Initialize environment with seed
        self.vecenv.async_reset(self.config.seed + self.rank)

        # Get the underlying metta grid environment
        self.metta_grid_env = self.vecenv.driver_env

        logger.info(
            f"Training environment setup: "
            f"num_envs={self.num_envs}, "
            f"batch_size={self.batch_size}, "
            f"target_batch_size={self.target_batch_size}, "
            f"num_agents={self.num_agents}"
        )

        return self.metta_grid_env, self.target_batch_size, self.batch_size, self.num_envs

    def step(self, actions: torch.Tensor) -> Experience:
        """Execute environment step with actions.

        Args:
            actions: Actions to execute

        Returns:
            Experience containing observations, rewards, etc.
        """
        if self.vecenv is None:
            raise RuntimeError("Environment not setup. Call setup() first.")

        # Send actions to environment
        self.vecenv.send_actions(actions)

        # Get observations and convert to experience
        obs = self.vecenv.get_observations()
        experience = Experience.from_vecenv_observations(obs)

        return experience

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the environment.

        Args:
            seed: Optional seed for reset
        """
        if self.vecenv is None:
            raise RuntimeError("Environment not setup. Call setup() first.")

        seed = seed if seed is not None else self.config.seed + self.rank
        self.vecenv.async_reset(seed)

    def close(self) -> None:
        """Close the environment."""
        if self.vecenv is not None:
            self.vecenv.close()
            self.vecenv = None

    def get_env(self) -> Any:
        """Get the underlying metta grid environment.

        Returns:
            The metta grid environment
        """
        if self.metta_grid_env is None:
            raise RuntimeError("Environment not setup. Call setup() first.")
        return self.metta_grid_env

    def get_vecenv(self) -> Any:
        """Get the vectorized environment.

        Returns:
            The vectorized environment
        """
        if self.vecenv is None:
            raise RuntimeError("Environment not setup. Call setup() first.")
        return self.vecenv

    def get_batch_info(self) -> Tuple[int, int, int]:
        """Get batch size information.

        Returns:
            Tuple of (target_batch_size, batch_size, num_envs)
        """
        if self.target_batch_size is None:
            raise RuntimeError("Environment not setup. Call setup() first.")
        return self.target_batch_size, self.batch_size, self.num_envs


def experience_generator(training_env: TrainingEnvironment, policy: Any) -> Any:
    """Generate experience by running policy in environment.

    This is a generator that yields experience from the training environment
    by executing the policy and collecting trajectories.

    Args:
        training_env: The training environment
        policy: The policy to execute

    Yields:
        Experience batches from environment interaction
    """
    while True:
        # Get current observation from environment
        obs = training_env.vecenv.get_observations()

        # Convert to experience format
        experience = Experience.from_vecenv_observations(obs)

        # Get actions from policy
        with torch.no_grad():
            actions = policy.get_actions(experience)

        # Step environment with actions
        training_env.vecenv.send_actions(actions)

        # Yield the experience
        yield experience
