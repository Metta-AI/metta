"""Core curriculum implementations and utilities."""

from __future__ import annotations

import abc
import random
from abc import ABC
from typing import Annotated, ClassVar, List, Optional, Union

import numpy as np
from pydantic import ConfigDict, Field, field_validator

from metta.common.util.config import Config
from metta.mettagrid.mettagrid_config import EnvConfig

from .task_generator import (
    BucketedTaskGeneratorConfig,
    SingleTaskGeneratorConfig,
    TaskGeneratorSetConfig,
)


class CurriculumTask:
    """A task instance with a task_id and env_cfg."""

    def __init__(self, task_id: int, env_cfg: EnvConfig):
        self._task_id = task_id
        self._env_cfg = env_cfg
        self._num_completions = 0
        self._total_score = 0.0
        self._mean_score = 0.0
        self._num_scheduled = 0

    def complete(self, score: float):
        """Notify curriculum that a task has been completed with given score."""
        self._num_completions += 1
        self._total_score += score
        self._mean_score = self._total_score / self._num_completions

    def get_env_cfg(self) -> EnvConfig:
        """Get the env_cfg for the task."""
        return self._env_cfg


class CurriculumAlgorithmHypers(Config, ABC):
    """Hyperparameters for the CurriculumAlgorithm."""

    initial_weights: Optional[List[float]] = None

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


class CurriculumAlgorithm(ABC):
    """Base class for curriculum algorithms that manage task sampling weights.

    Curriculum algorithms are responsible for:
    1. Maintaining weights for each child task
    2. Updating weights based on task completion feedback
    3. Providing normalized probabilities for sampling

    The Curriculum will use these algorithms to decide which child to sample next.
    """

    num_tasks: int
    weights: np.ndarray
    probabilities: np.ndarray
    hypers: CurriculumAlgorithmHypers

    # API that Curriculum uses

    def update(self, child_idx: int, score: float) -> None:
        """Update weights in-place based on task completion."""
        self._update_weights(child_idx, score)
        self._update_probabilities()

    def sample_idx(self) -> int:
        """Sample a child index based on current probabilities."""
        return np.random.choice(len(self.probabilities), p=self.probabilities)

    # Subclass methods to override

    def __init__(self, num_tasks: int, hypers: Optional[CurriculumAlgorithmHypers] = None):
        if num_tasks <= 0:
            raise ValueError(f"Number of tasks must be positive. num_tasks {num_tasks}")
        self.num_tasks = num_tasks

        if hypers is None:
            hypers = DiscreteRandomHypers()
        self.hypers = hypers

        if hypers.initial_weights is None:
            self.weights = np.ones(num_tasks, dtype=np.float32)
        else:
            self.weights = np.array(hypers.initial_weights, dtype=np.float32)
            if len(self.weights) != num_tasks:
                raise ValueError(
                    f"Initial weights must have length {num_tasks}. weights {self.weights} length: {len(self.weights)}"
                )

        self._update_probabilities()

    def stats(self, prefix: str = "") -> dict[str, float]:
        """Return statistics for logging purposes. Add `prefix` to all keys."""
        return {}

    @abc.abstractmethod
    def _update_weights(self, child_idx: int, score: float) -> None:
        """Logic for updating weights in-place based on task completion goes here."""
        pass

    # Helper methods

    def _update_probabilities(self):
        """Update the probability distribution based on current weights."""
        assert len(self.weights) == self.num_tasks, (
            f"Weights must have length {self.num_tasks}. weights {self.weights} length: {len(self.weights)}"
        )
        assert self.weights.sum() > 0, f"Weights must be non-zero-sum. weights {self.weights} sum: {self.weights.sum()}"
        assert np.all(self.weights >= 0), f"Weights must be non-negative. weights {self.weights}"
        self.probabilities = self.weights / self.weights.sum()


class DiscreteRandomHypers(CurriculumAlgorithmHypers):
    """Hyperparameters for DiscreteRandomCurriculum."""

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


class CurriculumConfig(Config):
    """Base configuration for Curriculum."""

    task_generator_config: Annotated[
        Union[SingleTaskGeneratorConfig, TaskGeneratorSetConfig, BucketedTaskGeneratorConfig],
        Field(discriminator="type", description="TaskGenerator configuration"),
    ]
    max_task_id: int = Field(default=1000000, gt=0, description="Maximum task id to generate")

    num_active_tasks: int = Field(default=100, gt=0, description="Number of active tasks to maintain")
    new_task_rate: float = Field(default=0.01, ge=0, le=1.0, description="Rate of new tasks to generate")

    # Algorithm configuration
    algorithm_hypers: Optional[CurriculumAlgorithmHypers] = Field(
        default=None, description="Curriculum algorithm hyperparameters"
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    @field_validator("num_active_tasks")
    @classmethod
    def validate_num_active_tasks(cls, v, info):
        max_task_id = info.data.get("max_task_id", 1000000)
        if v > max_task_id:
            raise ValueError("num_active_tasks must be less than max_task_id")
        return v

    def make(self) -> Curriculum:
        """Make a Curriculum from this configuration."""
        return Curriculum(self)


class Curriculum:
    """Base curriculum class that uses TaskGenerator to generate EnvConfigs and returns Tasks.

    Curriculum takes a CurriculumConfig, and supports get_task(). It uses the task generator
    to generate the EnvConfig and then returns a Task(env_cfg). It can optionally use a
    CurriculumAlgorithm for intelligent task selection.
    """

    def __init__(self, config: CurriculumConfig, seed: int = 0):
        self._config = config
        self._task_generator = config.task_generator_config.create()
        self._rng = random.Random(seed)
        self._tasks: dict[int, CurriculumTask] = {}
        self._task_ids: set[int] = set()
        self._num_created = 0
        self._num_evicted = 0

        # Initialize curriculum algorithm if provided
        self._algorithm: Optional[CurriculumAlgorithm] = None
        if config.algorithm_hypers is not None:
            self._algorithm = config.algorithm_hypers.create(config.num_active_tasks)

    def get_task(self) -> CurriculumTask:
        """Sample a task from the population."""
        if len(self._tasks) < self._config.num_active_tasks:
            task = self._create_task()
        elif self._rng.random() < self._config.new_task_rate:
            self._evict_task()
            task = self._create_task()
        else:
            task = self._choose_task()

        task._num_scheduled += 1
        return task

    def _choose_task(self) -> CurriculumTask:
        """Choose a task from the population."""
        if self._algorithm is not None and len(self._tasks) > 0:
            # Use algorithm for task selection
            task_list = list(self._tasks.values())
            if len(task_list) == self._algorithm.num_tasks:
                # Algorithm has correct number of tasks, use it
                task_idx = self._algorithm.sample_idx()
                return task_list[task_idx]
            else:
                # Algorithm doesn't match current task count, fall back to random
                return self._tasks[self._rng.choice(list(self._tasks.keys()))]
        else:
            # No algorithm or no tasks, use random selection
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
        task_id = self._rng.choice(list(self._task_ids))
        self._task_ids.remove(task_id)
        self._tasks.pop(task_id)
        self._num_evicted += 1

    def update_task_performance(self, task_id: int, score: float):
        """Update the curriculum algorithm with task performance."""
        if self._algorithm is not None:
            # Find the task index in the algorithm's task list
            task_list = list(self._tasks.values())
            for i, task in enumerate(task_list):
                if task._task_id == task_id:
                    self._algorithm.update(i, score)
                    break

    def stats(self) -> dict:
        """Return curriculum statistics for logging purposes."""
        base_stats = {
            "num_created": self._num_created,
            "num_evicted": self._num_evicted,
            "num_completed": sum(task._num_completions for task in self._tasks.values()),
            "num_scheduled": sum(task._num_scheduled for task in self._tasks.values()),
            "num_active_tasks": len(self._tasks),
        }

        # Add algorithm statistics if available
        if self._algorithm is not None:
            algorithm_stats = self._algorithm.stats(prefix="algorithm/")
            base_stats = {**base_stats, **algorithm_stats}

        return base_stats
