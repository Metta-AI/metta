"""Core curriculum implementations and utilities."""

from __future__ import annotations

import abc
import random
from abc import ABC
from typing import ClassVar, Dict, List, Optional, Tuple

import numpy as np
from pydantic import ConfigDict, Field, field_validator

from metta.cogworks.curriculum.task_generator import AnyTaskGeneratorConfig
from metta.common.config import Config
from metta.mettagrid.mettagrid_config import EnvConfig


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

    type: str = Field(description="Type of algorithm hyperparameters")
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
    hypers: CurriculumAlgorithmHypers

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
        self, num_tasks: int, hypers: Optional[CurriculumAlgorithmHypers] = None, initialize_weights: bool = True
    ):
        if num_tasks <= 0:
            raise ValueError(f"Number of tasks must be positive. num_tasks {num_tasks}")
        self.num_tasks = num_tasks

        if hypers is None:
            hypers = DiscreteRandomHypers()
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


class DiscreteRandomHypers(CurriculumAlgorithmHypers):
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


class LearningProgressHypers(CurriculumAlgorithmHypers):
    """Hyperparameters for LearningProgressCurriculum."""

    type: str = "learning_progress"
    num_tasks: int = Field(default=1000, description="Number of tasks to maintain in memory")
    sample_size: int = Field(default=10, description="Number of tasks to sample (K)")
    max_samples: int = Field(default=20, description="Maximum samples before eviction (A)")
    exploration_weight: float = Field(default=0.1, description="Weight for exploration vs exploitation")

    def algorithm_type(self) -> str:
        return "learning_progress"

    def create(self, num_tasks: int) -> "CurriculumAlgorithm":
        return LearningProgressCurriculum(num_tasks, self)


class LearningProgressCurriculum(CurriculumAlgorithm):
    """Learning progress curriculum with local task memory.

    Maintains N tasks locally with seed, task family, sample count, and score.
    Samples K tasks and balances exploration/exploitation for selection.
    Evicts tasks sampled more than A times with low scores.
    """

    def __init__(self, num_tasks: int, hypers: Optional[CurriculumAlgorithmHypers] = None):
        if hypers is None:
            hypers = LearningProgressHypers()
        # Don't initialize weights since this algorithm uses its own sampling strategy
        super().__init__(num_tasks, hypers, initialize_weights=False)

        # Local task memory: {task_id: (seed, family, sample_count, current_score, recent_score)}
        self._task_memory: Dict[int, Tuple[int, str, int, float, float]] = {}
        self._task_ids: List[int] = []
        self._current_task_list: List = []

    def add_task(self, task_id: int, seed: int, family: str):
        """Add a new task to local memory."""
        hypers = self.hypers
        if not isinstance(hypers, LearningProgressHypers):
            return

        if len(self._task_memory) >= hypers.num_tasks:
            # Evict oldest task if at capacity
            oldest_id = self._task_ids.pop(0)
            del self._task_memory[oldest_id]

        self._task_memory[task_id] = (seed, family, 0, 0.0, 0.0)
        self._task_ids.append(task_id)

    def sample_task_id(self) -> int:
        """Sample K tasks and select one using exploration/exploitation balance."""
        if len(self._task_memory) == 0:
            raise ValueError("No tasks in memory")

        hypers = self.hypers
        if not isinstance(hypers, LearningProgressHypers):
            raise ValueError("Expected LearningProgressHypers")

        # Sample K tasks
        sample_size = min(hypers.sample_size, len(self._task_memory))
        candidate_ids = np.random.choice(list(self._task_memory.keys()), size=sample_size, replace=False)

        # Calculate scores balancing exploration and exploitation
        scores = []
        for task_id in candidate_ids:
            seed, family, sample_count, current_score, recent_score = self._task_memory[task_id]

            # Combined score: current + recent
            combined_score = current_score + recent_score

            # Exploration bonus for less sampled tasks
            exploration_bonus = hypers.exploration_weight / (sample_count + 1)

            scores.append(combined_score + exploration_bonus)

        # Select task with highest score
        best_idx = np.argmax(scores)
        return candidate_ids[best_idx]

    def should_evict(self, task_id: int) -> bool:
        """Check if task should be evicted (sampled > A times with low score)."""
        if task_id not in self._task_memory:
            return False

        hypers = self.hypers
        if not isinstance(hypers, LearningProgressHypers):
            return False

        seed, family, sample_count, current_score, recent_score = self._task_memory[task_id]
        combined_score = current_score + recent_score

        return sample_count >= hypers.max_samples and combined_score < 0.5

    def evict_task(self, task_id: int):
        """Remove task from memory."""
        if task_id in self._task_memory:
            del self._task_memory[task_id]
            self._task_ids.remove(task_id)

    def _update_weights(self, child_idx: int, score: float) -> None:
        """Update task scores in memory."""
        # Find task by index in current task list
        if hasattr(self, "_current_task_list") and self._current_task_list:
            task_list = self._current_task_list
            if child_idx < len(task_list):
                task = task_list[child_idx]
                task_id = task._task_id

                if task_id in self._task_memory:
                    seed, family, sample_count, current_score, recent_score = self._task_memory[task_id]

                    # Update scores: recent becomes current, new score becomes recent
                    self._task_memory[task_id] = (seed, family, sample_count + 1, recent_score, score)

    def stats(self, prefix: str = "") -> dict[str, float]:
        """Return learning progress statistics."""
        if not self._task_memory:
            return {}

        scores = [float(current + recent) for _, _, _, current, recent in self._task_memory.values()]
        sample_counts = [float(count) for _, _, count, _, _ in self._task_memory.values()]

        return {
            f"{prefix}num_tasks_in_memory": float(len(self._task_memory)),
            f"{prefix}avg_score": float(np.mean(scores)),
            f"{prefix}avg_sample_count": float(np.mean(sample_counts)),
            f"{prefix}max_sample_count": float(max(sample_counts)),
        }


class CurriculumConfig(Config):
    """Base configuration for Curriculum."""

    task_generator: AnyTaskGeneratorConfig = Field(description="TaskGenerator configuration")
    max_task_id: int = Field(default=1000000, gt=0, description="Maximum task id to generate")

    num_active_tasks: int = Field(default=10000, gt=0, description="Number of active tasks to maintain")
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
        max_task_id = info.data["max_task_id"]
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
        self._task_generator = config.task_generator.create()
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

            # Update algorithm's current task list for learning progress
            if isinstance(self._algorithm, LearningProgressCurriculum):
                self._algorithm._current_task_list = task_list

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

        # Add task to learning progress memory if using that algorithm
        if isinstance(self._algorithm, LearningProgressCurriculum):
            # Extract seed and family from task generator (this may need adjustment based on actual implementation)
            seed = task_id  # Use task_id as seed for now
            family = "default"  # Default family, could be extracted from env_cfg
            self._algorithm.add_task(task_id, seed, family)

        return task

    def _evict_task(self):
        """Evict a task from the population."""
        # Check if learning progress algorithm wants to evict specific tasks
        if isinstance(self._algorithm, LearningProgressCurriculum):
            for task_id in list(self._task_ids):
                if self._algorithm.should_evict(task_id):
                    self._task_ids.remove(task_id)
                    self._tasks.pop(task_id)
                    self._algorithm.evict_task(task_id)
                    self._num_evicted += 1
                    return

        # Fall back to random eviction
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
