"""Core curriculum implementations and utilities."""

from __future__ import annotations

import abc
import random
import time
from abc import ABC
from collections import deque
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig

from pydantic import ConfigDict, Field

from metta.cogworks.curriculum.task_generator import AnyTaskGeneratorConfig, SingleTaskGeneratorConfig
from metta.mettagrid.config import Config
from metta.mettagrid.mettagrid_config import MettaGridConfig


class TaskTracker:
    """Tracks task metadata, performance history, and completion statistics."""

    def __init__(self, max_memory_tasks: int = 1000):
        self.max_memory_tasks = max_memory_tasks

        # Task memory: task_id -> (creation_time, completion_count, total_score, last_score)
        self._task_memory: Dict[int, Tuple[float, int, float, float]] = {}

        # Task creation order for efficient cleanup
        self._task_creation_order = deque()  # (timestamp, task_id) pairs

        # Performance tracking
        self._completion_history = deque(maxlen=1000)  # Recent completion scores

        # Cached values to avoid expensive recomputation
        self._cached_total_completions = 0
        self._cache_valid = False

    def track_task_creation(self, task_id: int) -> None:
        """Track when a task is created."""
        timestamp = time.time()
        self._task_memory[task_id] = (timestamp, 0, 0.0, 0.0)
        self._task_creation_order.append((timestamp, task_id))

        # Cleanup old tasks if we exceed memory limit
        if len(self._task_memory) > self.max_memory_tasks:
            self._cleanup_old_tasks()

        # Invalidate cache when task structure changes
        self._cache_valid = False

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance statistics."""
        if task_id not in self._task_memory:
            self.track_task_creation(task_id)

        creation_time, completion_count, total_score, _ = self._task_memory[task_id]
        new_completion_count = completion_count + 1
        new_total_score = total_score + score

        self._task_memory[task_id] = (creation_time, new_completion_count, new_total_score, score)
        self._completion_history.append(score)

        # Update cached total completions incrementally
        if self._cache_valid:
            self._cached_total_completions += 1
        else:
            self._cache_valid = False

    def get_task_stats(self, task_id: int) -> Optional[Dict[str, float]]:
        """Get statistics for a specific task."""
        if task_id not in self._task_memory:
            return None

        creation_time, completion_count, total_score, last_score = self._task_memory[task_id]

        if completion_count == 0:
            return {
                "completion_count": 0,
                "mean_score": 0.0,
                "last_score": 0.0,
                "age_seconds": time.time() - creation_time,
            }

        return {
            "completion_count": completion_count,
            "mean_score": total_score / completion_count,
            "last_score": last_score,
            "age_seconds": time.time() - creation_time,
        }

    def get_all_tracked_tasks(self) -> list[int]:
        """Get all currently tracked task IDs."""
        return list(self._task_memory.keys())

    def remove_task(self, task_id: int) -> None:
        """Remove a task from tracking."""
        if task_id in self._task_memory:
            self._task_memory.pop(task_id)
            # Remove from creation order (expensive but infrequent)
            self._task_creation_order = deque((ts, tid) for ts, tid in self._task_creation_order if tid != task_id)
            self._cache_valid = False

    def _cleanup_old_tasks(self) -> None:
        """Remove oldest tasks to keep memory usage under control."""
        while len(self._task_memory) > self.max_memory_tasks and self._task_creation_order:
            _, old_task_id = self._task_creation_order.popleft()
            if old_task_id in self._task_memory:
                self._task_memory.pop(old_task_id)

        # Cache may still be valid after cleanup if we tracked changes
        self._cache_valid = False

    def get_global_stats(self) -> Dict[str, float]:
        """Get global task tracking statistics."""
        if not self._cache_valid:
            self._cached_total_completions = sum(
                completion_count for _, completion_count, _, _ in self._task_memory.values()
            )
            self._cache_valid = True

        return {
            "total_tracked_tasks": float(len(self._task_memory)),
            "total_completions": float(self._cached_total_completions),
            "avg_completions_per_task": float(self._cached_total_completions / len(self._task_memory))
            if self._task_memory
            else 0.0,
            "recent_completion_history_size": float(len(self._completion_history)),
        }


def get_algorithm_hypers_discriminator(v):
    """Discriminator function for algorithm hypers types."""
    if isinstance(v, dict) and "type" in v:
        return v["type"]
    return None


class CurriculumTask:
    """A task instance with a task_id and env_cfg."""

    def __init__(self, task_id: int, env_cfg, bucket_values: Optional[Dict[str, Any]] = None):
        self._task_id = task_id
        self._env_cfg = env_cfg
        self._bucket_values = bucket_values or {}
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

    def get_bucket_values(self):
        """Get the bucket values that were used to generate this task."""
        return self._bucket_values


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
        return DiscreteRandomCurriculum(num_tasks, self)

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )


class CurriculumAlgorithm(ABC):
    """
    Curriculum algorithms are responsible for:
    1. Scoring tasks based on their learning progress or other metrics
    2. Recommending which tasks to evict when the pool is full
    3. Tracking task performance for algorithm-specific purposes
    4. Providing feedback to Curriculum for task selection

    The Curriculum maintains the task pool and lifecycle, while algorithms provide guidance.
    """

    num_tasks: int
    hypers: CurriculumAlgorithmConfig

    # Core API for task scoring and recommendations

    @abc.abstractmethod
    def score_tasks(self, task_ids: List[int]) -> Dict[int, float]:
        """Score tasks for selection purposes. Higher scores = more likely to be selected."""
        pass

    @abc.abstractmethod
    def recommend_eviction(self, task_ids: List[int]) -> Optional[int]:
        """Recommend which task to evict. Return None for random selection."""
        pass

    def on_task_evicted(self, task_id: int) -> None:
        """Notification that a task has been evicted from the pool."""
        # Default implementation removes from task tracker
        if hasattr(self, "task_tracker"):
            self.task_tracker.remove_task(task_id)

    def on_task_created(self, task: "CurriculumTask") -> None:
        """Notification that a new task has been created."""
        # Default implementation tracks task creation
        if hasattr(self, "task_tracker"):
            self.task_tracker.track_task_creation(task._task_id)

    def update_task_performance(self, task_id: int, score: float):
        """Update task performance. Default implementation updates task tracker."""
        if hasattr(self, "task_tracker"):
            self.task_tracker.update_task_performance(task_id, score)

    def should_evict_task(self, task_id: int, min_presentations: int = 5) -> bool:
        """Check if a task should be evicted based on algorithm-specific criteria.

        Default implementation uses task_tracker to check minimum presentations.
        Subclasses should override to implement their own eviction criteria.

        Args:
            task_id: The task to check
            min_presentations: Minimum number of task presentations before eviction

        Returns:
            True if task should be evicted
        """
        if hasattr(self, "task_tracker"):
            task_stats = self.task_tracker.get_task_stats(task_id)
            if not task_stats:
                return False
            # Only evict tasks with sufficient presentations
            return task_stats["completion_count"] >= min_presentations
        return False

    def __init__(
        self, num_tasks: int, hypers: Optional[CurriculumAlgorithmConfig] = None, initialize_weights: bool = True
    ):
        if num_tasks <= 0:
            raise ValueError(f"Number of tasks must be positive. num_tasks {num_tasks}")
        self.num_tasks = num_tasks

        if hypers is None:
            hypers = DiscreteRandomConfig()
        self.hypers = hypers

        # Initialize task tracker for algorithms that need it
        # Can be overridden in subclasses
        self.task_tracker = TaskTracker(max_memory_tasks=1000)

    def stats(self, prefix: str = "") -> dict[str, float]:
        """Return statistics for logging purposes. Add `prefix` to all keys."""
        stats = {}
        if hasattr(self, "task_tracker"):
            tracker_stats = self.task_tracker.get_global_stats()
            stats.update({f"{prefix}tracker/{k}": v for k, v in tracker_stats.items()})
        return stats

    def get_task_from_pool(self, task_generator, rng) -> "CurriculumTask":
        """Get a task from the pool. Default implementation creates a simple task."""

        task_id = rng.randint(0, 1000000)
        env_cfg = task_generator.get_task(task_id)
        return CurriculumTask(task_id, env_cfg)


class DiscreteRandomConfig(CurriculumAlgorithmConfig):
    """Hyperparameters for DiscreteRandomCurriculum."""

    type: str = "discrete_random"

    def algorithm_type(self) -> str:
        return "discrete_random"


class DiscreteRandomCurriculum(CurriculumAlgorithm):
    """Curriculum algorithm that samples from a discrete distribution of weights.

    A named class for the simplest case where weights don't change based on
    task performance.
    """

    def score_tasks(self, task_ids: List[int]) -> Dict[int, float]:
        """All tasks have equal score for random selection."""
        return {task_id: 1.0 for task_id in task_ids}

    def recommend_eviction(self, task_ids: List[int]) -> Optional[int]:
        """No preference for eviction - let Curriculum choose randomly."""
        return None


class CurriculumConfig(Config):
    """Base configuration for Curriculum."""

    task_generator: AnyTaskGeneratorConfig = Field(description="TaskGenerator configuration")
    max_task_id: int = Field(default=1000000, gt=0, description="Maximum task ID to generate")
    num_active_tasks: int = Field(default=10000, gt=0, description="Number of active tasks to maintain")

    # Curriculum behavior options
    min_presentations_for_eviction: int = Field(
        default=5, gt=0, description="Minimum task presentations before eviction"
    )

    algorithm_config: Optional[Union["DiscreteRandomConfig", "LearningProgressConfig"]] = Field(
        default=None, description="Curriculum algorithm hyperparameters"
    )

    @classmethod
    def from_mg(cls, mg_config: MettaGridConfig) -> "CurriculumConfig":
        """Create a CurriculumConfig from a MettaGridConfig."""
        return cls(
            task_generator=SingleTaskGeneratorConfig(env=mg_config),
        )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    def model_post_init(self, __context) -> None:
        """Validate configuration after initialization."""
        super().model_post_init(__context)

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
    to generate the EnvConfig and then returns a Task(env_cfg). It always uses a
    CurriculumAlgorithm for task selection (defaults to DiscreteRandom if none specified).
    """

    def __init__(self, config: CurriculumConfig, seed: int = 0):
        self._config = config
        self._task_generator = config.task_generator.create()
        self._rng = random.Random(seed)
        self._tasks: dict[int, CurriculumTask] = {}
        self._num_created = 0
        self._num_evicted = 0

        # Always have an algorithm - default to DiscreteRandom
        algorithm_config = config.algorithm_config or DiscreteRandomConfig()
        self._algorithm = algorithm_config.create(config.num_active_tasks)

        # Pass curriculum reference to algorithm for stats updates
        if hasattr(self._algorithm, "set_curriculum_reference"):
            self._algorithm.set_curriculum_reference(self)

        # Always initialize task pool at capacity
        self._initialize_at_capacity()

    def get_task(self) -> CurriculumTask:
        """Sample a task from the population."""
        # Curriculum always manages the task pool - no delegation
        if len(self._tasks) < self._config.num_active_tasks:
            task = self._create_task()
        else:
            task = self._choose_task()

            # Use algorithm criteria for eviction
            evictable_tasks = [
                tid
                for tid in self._tasks.keys()
                if self._algorithm.should_evict_task(tid, self._config.min_presentations_for_eviction)
            ]
            if evictable_tasks:
                # Evict a task that meets the criteria and create a new one
                evict_candidate = self._algorithm.recommend_eviction(evictable_tasks)
                if evict_candidate is not None:
                    self._evict_specific_task(evict_candidate)
                    task = self._create_task()

        task._num_scheduled += 1
        return task

    def _initialize_at_capacity(self) -> None:
        """Initialize the task pool to full capacity."""
        while len(self._tasks) < self._config.num_active_tasks:
            self._create_task()

    def _evict_specific_task(self, task_id: int) -> None:
        """Evict a specific task by ID."""
        if task_id not in self._tasks:
            return

        # Notify algorithm of eviction
        self._algorithm.on_task_evicted(task_id)

        self._tasks.pop(task_id)
        self._num_evicted += 1

    def _choose_task(self) -> CurriculumTask:
        """Choose a task from the population using algorithm guidance."""
        # Get algorithm's task selection preferences
        task_scores = self._algorithm.score_tasks(list(self._tasks.keys()))
        if task_scores:
            # Convert scores to probabilities for sampling
            task_ids = list(task_scores.keys())
            scores = list(task_scores.values())
            total_score = sum(scores)
            if total_score > 0:
                probabilities = [score / total_score for score in scores]
                selected_id = self._rng.choices(task_ids, weights=probabilities)[0]
                return self._tasks[selected_id]

        # Fallback to random selection if no scores provided
        return self._tasks[self._rng.choice(list(self._tasks.keys()))]

    def _create_task(self) -> CurriculumTask:
        """Create a new task."""
        task_id = self._rng.randint(0, self._config.max_task_id)
        while task_id in self._tasks:
            task_id = self._rng.randint(0, self._config.max_task_id)
        env_cfg = self._task_generator.get_task(task_id)

        # Extract bucket values if available
        bucket_values = {}
        if hasattr(self._task_generator, "_last_bucket_values"):
            bucket_values = self._task_generator._last_bucket_values.copy()

        task = CurriculumTask(task_id, env_cfg, bucket_values)
        self._tasks[task_id] = task
        self._num_created += 1

        # Notify algorithm of new task
        if hasattr(self._algorithm, "on_task_created"):
            self._algorithm.on_task_created(task)

        return task

    def update_task_performance(self, task_id: int, score: float):
        """Update the curriculum algorithm with task performance."""
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

        # Always add algorithm stats
        algorithm_stats = self._algorithm.stats()
        # Add algorithm prefix to avoid conflicts
        prefixed_algorithm_stats = {f"algorithm/{k}": v for k, v in algorithm_stats.items()}
        return {**base_stats, **prefixed_algorithm_stats}
