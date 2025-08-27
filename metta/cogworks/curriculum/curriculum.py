"""Core curriculum implementations and utilities."""

from __future__ import annotations

import abc
import random
from abc import ABC
from typing import ClassVar, Optional, Union

import numpy as np
from pydantic import ConfigDict, Field

from metta.cogworks.curriculum.task_generator import AnyTaskGeneratorConfig
from metta.common.config import Config


def get_algorithm_hypers_discriminator(v):
    """Discriminator function for algorithm hypers types."""
    if isinstance(v, dict) and "type" in v:
        return v["type"]
    return None


class CurriculumTask:
    """A task instance with a task_id and env_cfg."""

    def __init__(self, task_id: int, env_cfg):
        self._task_id = task_id
        self._env_cfg = env_cfg
        self._num_completions = 0
        self._total_score = 0.0
        self._mean_score = 0.0
        self._num_scheduled = 0

    def complete(self, score: float):
        """Complete the task with a score."""
        self._num_completions += 1
        self._total_score += score
        self._mean_score = self._total_score / self._num_completions

    def get_env_cfg(self):
        """Get the environment configuration for this task."""
        return self._env_cfg


class CurriculumAlgorithmConfig(Config, ABC):
    """Hyperparameters for the CurriculumAlgorithm."""

    type: str = Field(description="Type of algorithm hyperparameters")
    initial_weights: Optional[list[float]] = None

    @abc.abstractmethod
    def algorithm_type(self) -> str:
        """Return the algorithm type string used in configs."""
        pass

    def create(self, num_tasks: int) -> "CurriculumAlgorithm":
        """Create the curriculum algorithm with these hyperparameters.

        Args:
            num_tasks: Number of tasks the algorithm will manage

        Returns:
            Configured curriculum algorithm instance
        """
        # The default implementation is to use DiscreteRandomCurriculum
        return DiscreteRandomCurriculum(num_tasks, self)

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )


class CurriculumAlgorithm(ABC):
    """
    Curriculum algorithms are responsible for:
    1. Maintaining weights for each child task (optional)
    2. Updating weights based on task completion feedback (optional)
    3. Providing normalized probabilities for sampling

    The Curriculum will use these algorithms to decide which child to sample next.
    """

    num_tasks: int
    weights: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    hypers: CurriculumAlgorithmConfig

    # API that Curriculum uses

    def update(self, child_idx: int, score: float) -> None:
        """Update weights in-place based on task completion."""
        self._update_weights(child_idx, score)
        if self.weights is not None:
            self._update_probabilities()

    def sample_idx(self) -> int:
        """Sample a child index based on current probabilities."""
        if self.probabilities is not None:
            return np.random.choice(len(self.probabilities), p=self.probabilities)
        else:
            # Fallback to uniform random if no probabilities available
            return np.random.choice(self.num_tasks)

    # Subclass methods to override

    def __init__(
        self, num_tasks: int, hypers: Optional[CurriculumAlgorithmConfig] = None, initialize_weights: bool = True
    ):
        if num_tasks <= 0:
            raise ValueError(f"Number of tasks must be positive. num_tasks {num_tasks}")
        self.num_tasks = num_tasks

        if hypers is None:
            hypers = DiscreteRandomConfig()
        self.hypers = hypers

        # Initialize weights only if requested and algorithm uses them
        if initialize_weights:
            if hypers.initial_weights is None:
                self.weights = np.ones(num_tasks, dtype=np.float32)
            else:
                self.weights = np.array(hypers.initial_weights, dtype=np.float32)
                if len(self.weights) != num_tasks:
                    raise ValueError(
                        f"Initial weights must have length {num_tasks}. "
                        f"weights {self.weights} length: {len(self.weights)}"
                    )
            self._update_probabilities()

    def stats(self, prefix: str = "") -> dict[str, float]:
        """Return statistics for logging purposes. Add `prefix` to all keys."""
        return {}

    @abc.abstractmethod
    def _update_weights(self, child_idx: int, score: float) -> None:
        """Update weights based on task completion. Override in subclasses that use weights."""
        pass

    def get_task_from_pool(self, task_generator, rng) -> "CurriculumTask":
        """Get a task from the pool. Default implementation creates a simple task."""

        # Generate a task ID
        task_id = rng.randint(0, 1000000)

        # Get environment configuration
        env_cfg = task_generator.get_task(task_id)

        # Create and return task
        return CurriculumTask(task_id, env_cfg)

    @abc.abstractmethod
    def update_task_performance(self, task_id: int, score: float):
        """Update task performance. Override in subclasses that track performance."""
        pass

    # Helper methods

    def _update_probabilities(self):
        """Update the probability distribution based on current weights."""
        if self.weights is None:
            return

        assert len(self.weights) == self.num_tasks, (
            f"Weights must have length {self.num_tasks}. weights {self.weights} length: {len(self.weights)}"
        )
        assert self.weights.sum() > 0, f"Weights must be non-zero-sum. weights {self.weights} sum: {self.weights.sum()}"
        assert np.all(self.weights >= 0), f"Weights must be non-negative. weights {self.weights}"
        self.probabilities = self.weights / self.weights.sum()


class DiscreteRandomConfig(CurriculumAlgorithmConfig):
    """Hyperparameters for DiscreteRandomCurriculum."""

    type: str = "discrete_random"

    def algorithm_type(self) -> str:
        return "discrete_random"


class DiscreteRandomCurriculum(CurriculumAlgorithm):
    """Curriculum algorithm that samples from a discrete distribution of weights.

    Already implemented by CurriculumAlgorithm base class - this just provides
    a named class for the simplest case where weights don't change based on
    task performance.
    """

    def _update_weights(self, child_idx: int, score: float) -> None:
        pass

    def update_task_performance(self, task_id: int, score: float):
        """Update task performance - no-op for discrete random curriculum."""
        pass


class CurriculumConfig(Config):
    """Base configuration for Curriculum."""

    task_generator: AnyTaskGeneratorConfig = Field(description="TaskGenerator configuration")
    max_task_id: int = Field(default=1000000, gt=0, description="Maximum task ID to generate")
    num_active_tasks: int = Field(default=10000, gt=0, description="Number of active tasks to maintain")
    new_task_rate: float = Field(default=0.01, ge=0, le=1.0, description="Rate of new tasks to generate")

    # Algorithm configuration
    algorithm_config: Optional[Union["DiscreteRandomConfig", "LearningProgressConfig"]] = Field(
        default=None, description="Curriculum algorithm hyperparameters"
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    def model_post_init(self, __context) -> None:
        """Validate configuration after initialization."""
        super().model_post_init(__context)

        # Validate that num_active_tasks doesn't exceed max_task_id
        if self.num_active_tasks > self.max_task_id:
            raise ValueError(
                f"num_active_tasks ({self.num_active_tasks}) cannot exceed max_task_id ({self.max_task_id})"
            )

    def make(self) -> "Curriculum":
        """Create a Curriculum from this configuration."""
        return Curriculum(self)


class Curriculum:
    """Base curriculum class that uses TaskGenerator to generate EnvConfigs and returns Tasks.

    Curriculum takes a CurriculumConfig, and supports get_task(). It uses the task generator
    to generate the EnvConfig and then returns a Task(env_cfg). It can optionally use a
    CurriculumAlgorithm for intelligent task selection.
    """

    def __init__(self, config: CurriculumConfig, seed: int = 0):
        self._config = config
        self._task_generator = config.task_generator.create()
        self._rng = random.Random(seed)
        self._tasks: dict[int, CurriculumTask] = {}
        self._task_ids: set[int] = set()
        self._num_created = 0
        self._num_evicted = 0

        # Initialize curriculum algorithm if provided
        self._algorithm: Optional[CurriculumAlgorithm] = None
        if config.algorithm_config is not None:
            self._algorithm = config.algorithm_config.create(config.num_active_tasks)

    def get_task(self) -> CurriculumTask:
        """Sample a task from the population."""
        if self._algorithm is not None:
            # Use algorithm's unified pool management
            return self._algorithm.get_task_from_pool(self._task_generator, self._rng)
        else:
            # Fallback to simple random selection
            if len(self._tasks) < self._config.num_active_tasks:
                task = self._create_task()
            else:
                task = self._choose_task()
                if self._rng.random() < self._config.new_task_rate:
                    self._evict_task()
                    task = self._create_task()

            task._num_scheduled += 1
            return task

    def _choose_task(self) -> CurriculumTask:
        """Choose a task from the population."""
        # Fallback to random selection when no algorithm
        return self._tasks[self._rng.choice(list(self._tasks.keys()))]

    def _create_task(self) -> CurriculumTask:
        """Create a new task."""
        task_id = self._rng.randint(0, self._config.max_task_id)
        while task_id in self._task_ids:
            task_id = self._rng.randint(0, self._config.max_task_id)
        self._task_ids.add(task_id)
        env_cfg = self._task_generator.get_task(task_id)

        task = CurriculumTask(task_id, env_cfg)
        self._tasks[task_id] = task
        self._num_created += 1

        return task

    def _evict_task(self):
        """Evict a task from the population."""
        # Fall back to random eviction when no algorithm
        task_id = self._rng.choice(list(self._task_ids))
        self._task_ids.remove(task_id)
        self._tasks.pop(task_id)
        self._num_evicted += 1

    def update_task_performance(self, task_id: int, score: float):
        """Update the curriculum algorithm with task performance."""
        if self._algorithm is not None:
            self._algorithm.update_task_performance(task_id, score)

    def stats(self) -> dict:
        """Return curriculum statistics for logging purposes."""
        # Always include basic curriculum stats
        base_stats = {
            "num_created": self._num_created,
            "num_evicted": self._num_evicted,
            "num_completed": sum(task._num_completions for task in self._tasks.values()),
            "num_scheduled": sum(task._num_scheduled for task in self._tasks.values()),
            "num_active_tasks": len(self._tasks),
        }

        if self._algorithm is not None:
            # Add algorithm stats with prefix
            algorithm_stats = self._algorithm.stats()
            # Add algorithm prefix to avoid conflicts
            prefixed_algorithm_stats = {f"algorithm/{k}": v for k, v in algorithm_stats.items()}
            return {**base_stats, **prefixed_algorithm_stats}
        else:
            return base_stats


# Import concrete config classes at the end to avoid circular imports
# ruff: noqa: E402
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
