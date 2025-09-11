from __future__ import annotations

import abc
import random
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig

from pydantic import ConfigDict, Field

from metta.cogworks.curriculum.task_generator import AnyTaskGeneratorConfig
from metta.mettagrid.config import Config


@dataclass
class TaskSample:
    """Complete task information for regeneration and performance tracking"""

    task_id: int
    score: float
    num_samples: int
    env_class: type
    seed: int
    bucket_values: Dict[str, Any] = field(default_factory=dict)

    def can_regenerate(self) -> bool:
        """Check if we have enough info to regenerate this task"""
        return self.env_class is not None and self.seed is not None

    def get_mean_score(self) -> float:
        """Get current mean performance"""
        return self.score / max(1, self.num_samples)


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

    type: str = Field(..., description="The type of the algorithm.")

    @abc.abstractmethod
    def algorithm_type(self) -> str:
        """Returns the type of the algorithm."""
        pass

    @abc.abstractmethod
    def create(self, num_tasks: int) -> "CurriculumAlgorithm":
        """Creates the algorithm with the given number of tasks."""
        pass

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
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

    def on_task_evicted(self, task_id: int) -> None:  # noqa: B027
        """Notification that a task has been evicted from the pool.

        Optional hook for algorithms to clean up task-specific data.
        Default implementation does nothing.
        """
        pass

    def on_task_created(self, task: "CurriculumTask") -> None:  # noqa: B027
        """Notification that a new task has been created.

        Optional hook for algorithms to initialize task tracking.
        Default implementation does nothing.
        """
        pass

    def update_task_performance(self, task_id: int, score: float):  # noqa: B027
        """Update task performance.

        Optional hook for algorithms to track task performance.
        Default implementation does nothing.
        """
        pass

    def should_evict_task(self, task_id: int, min_presentations: int = 5) -> bool:
        """Check if a task should be evicted based on criteria."""
        return False

    def stats(self, prefix: str = "") -> Dict[str, float]:
        """Get algorithm-specific statistics."""
        return {}

    def set_dependencies(self, task_pool, task_tracker) -> None:  # noqa: B027
        """Set dependencies from Curriculum.

        Optional hook for algorithms to receive references to task pool and tracker.
        Default implementation does nothing.
        """
        pass


class ScoreBasedEvictionPolicy:
    """Generic eviction policy based on score and sample count"""

    def __init__(self, min_samples: int = 5, bottom_percentile: float = 0.2):
        self.min_samples = min_samples
        self.bottom_percentile = bottom_percentile

    def should_evict_task(self, task: TaskSample) -> bool:
        """Check if task meets basic eviction criteria"""
        return task.num_samples >= self.min_samples

    def recommend_eviction(self, evictable_tasks: List[TaskSample]) -> Optional[int]:
        """Recommend task with lowest mean score for eviction"""
        if not evictable_tasks:
            return None

        # Sort by mean score (ascending)
        sorted_tasks = sorted(evictable_tasks, key=lambda t: t.get_mean_score())

        # Evict from bottom percentile
        threshold_index = max(0, int(len(sorted_tasks) * self.bottom_percentile))
        return sorted_tasks[threshold_index].task_id if sorted_tasks else None


class TaskPool:
    """Maintains a constantly full pool of TaskSamples"""

    def __init__(self, capacity: int, task_generator):
        self.capacity = capacity
        self.task_generator = task_generator
        self._tasks: Dict[int, TaskSample] = {}
        self._next_task_id = 0

        # Initialize to full capacity
        self._fill_to_capacity()

    def _fill_to_capacity(self) -> None:
        """Fill pool to capacity with new tasks"""
        while len(self._tasks) < self.capacity:
            self._add_new_task()

    def _add_new_task(self) -> TaskSample:
        """Generate and add a new task to the pool"""
        env_cfg = self.task_generator.get_task(self._next_task_id)
        bucket_values = getattr(env_cfg, "bucket_values", {})

        task_sample = TaskSample(
            task_id=self._next_task_id,
            score=0.0,
            num_samples=0,
            env_class=type(env_cfg),
            seed=getattr(env_cfg, "seed", None),
            bucket_values=bucket_values,
        )

        self._tasks[self._next_task_id] = task_sample
        self._next_task_id += 1
        return task_sample

    def evict_task(self, task_id: int) -> None:
        """Remove task and immediately add replacement"""
        if task_id in self._tasks:
            del self._tasks[task_id]
            self._add_new_task()  # Immediately replenish

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance statistics"""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.score += score
            task.num_samples += 1

    def get_all_tasks(self) -> List[TaskSample]:
        """Get all active tasks"""
        return list(self._tasks.values())

    def get_task(self, task_id: int) -> Optional[TaskSample]:
        """Get specific task by ID"""
        return self._tasks.get(task_id)

    def get_all_task_ids(self) -> List[int]:
        """Get all active task IDs"""
        return list(self._tasks.keys())


class DiscreteRandomConfig(CurriculumAlgorithmConfig):
    """Configuration for discrete random curriculum algorithm."""

    type: str = "discrete_random"

    def algorithm_type(self) -> str:
        return "discrete_random"

    def create(self, num_tasks: int) -> "DiscreteRandomCurriculum":
        return DiscreteRandomCurriculum(num_tasks, self)


class DiscreteRandomCurriculum(CurriculumAlgorithm):
    """Curriculum algorithm that samples from a discrete distribution of weights."""

    def __init__(self, num_tasks: int, hypers: DiscreteRandomConfig, initialize_weights: bool = True):
        self.num_tasks = num_tasks
        self.hypers = hypers

        # Initialize uniform weights
        if initialize_weights:
            self.weights = [1.0] * num_tasks
        else:
            self.weights = []

    def score_tasks(self, task_ids: List[int]) -> Dict[int, float]:
        """Return uniform scores for random selection."""
        return {task_id: 1.0 for task_id in task_ids}

    def recommend_eviction(self, task_ids: List[int]) -> Optional[int]:
        """Random eviction recommendation."""
        if not task_ids:
            return None
        return random.choice(task_ids)

    def should_evict_task(self, task_id: int, min_presentations: int = 5) -> bool:
        """Random curriculum doesn't evict based on performance."""
        return False


class CurriculumConfig(Config):
    """Configuration for the entire curriculum system."""

    task_generator: AnyTaskGeneratorConfig = Field(..., description="Task generator configuration")
    algorithm_config: Optional[Union[CurriculumAlgorithmConfig, "LearningProgressConfig"]] = Field(
        default=None, description="Curriculum algorithm configuration"
    )
    num_active_tasks: int = Field(default=10000, description="Number of tasks in the active pool")
    min_presentations_for_eviction: int = Field(default=5, description="Minimum presentations before eviction")

    # Backward compatibility field
    max_task_id: int = Field(default=1000000, description="Maximum task ID (for backward compatibility)")

    # Bucket logging configuration
    enable_detailed_bucket_logging: bool = Field(default=False, description="Enable detailed bucket logging")
    max_memory_tasks: int = Field(default=1000, description="Maximum tasks to track in memory")
    max_bucket_axes: int = Field(default=3, description="Maximum bucket axes to track")

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,  # Allow modification for tests
        populate_by_name=True,
    )

    @classmethod
    def from_mg(cls, mg_config) -> "CurriculumConfig":
        """Create a CurriculumConfig from an MettaGridConfig."""
        from .task_generator import SingleTaskGeneratorConfig

        task_gen_config = SingleTaskGeneratorConfig(env=mg_config)
        return cls(task_generator=task_gen_config)

    def model_post_init(self, __context) -> None:
        """Validate configuration after initialization."""
        if self.num_active_tasks > self.max_task_id:
            raise ValueError(
                f"num_active_tasks ({self.num_active_tasks}) cannot be greater than max_task_id ({self.max_task_id})"
            )

    def make(self) -> "Curriculum":
        return Curriculum(self)


class Curriculum:
    """Coordinates task pool, algorithm, and tracking"""

    def __init__(self, config: CurriculumConfig, seed: int = 0):
        self._config = config
        self._rng = random.Random(seed)

        # Always-full task pool
        self.task_pool = TaskPool(config.num_active_tasks, config.task_generator.create())

        # Optional bucket tracking
        self.task_tracker = None
        if config.enable_detailed_bucket_logging:
            from .task_tracker import TaskTracker

            self.task_tracker = TaskTracker(
                max_memory_tasks=config.max_memory_tasks,
                max_bucket_axes=config.max_bucket_axes,
                enable_detailed_bucket_logging=True,
            )

        # Algorithm with dependencies
        if config.algorithm_config is None:
            self.algorithm = DiscreteRandomCurriculum(config.num_active_tasks, DiscreteRandomConfig())
        else:
            self.algorithm = config.algorithm_config.create(config.num_active_tasks)

        if hasattr(self.algorithm, "set_dependencies"):
            self.algorithm.set_dependencies(self.task_pool, self.task_tracker)

        # Backwards compatibility attributes
        self._algorithm = self.algorithm
        self._task_generator = config.task_generator.create()
        self._tasks = {}  # Will be populated lazily
        self._num_created = config.num_active_tasks  # Start at capacity
        self._num_evicted = 0

        # Initialize _tasks dict for backwards compatibility
        self._initialize_tasks_dict()

    def get_task(self) -> CurriculumTask:
        """Get task using algorithm scoring"""
        tasks = self.task_pool.get_all_tasks()
        task_ids = [t.task_id for t in tasks]

        # Score and sample
        scores = self.algorithm.score_tasks(task_ids)
        selected_task_id = self._sample_by_scores(scores)

        # Handle eviction (but don't evict the selected task)
        evictable_tasks = []
        for task_id in task_ids:
            # Don't evict the task we just selected
            if task_id == selected_task_id:
                continue
            task = self.task_pool.get_task(task_id)
            if task and hasattr(self.algorithm, "eviction_policy"):
                if self.algorithm.eviction_policy.should_evict_task(task):
                    evictable_tasks.append(task)

        if evictable_tasks:
            evict_id = self.algorithm.recommend_eviction([t.task_id for t in evictable_tasks])
            if evict_id and evict_id != selected_task_id:  # Double check we're not evicting selected task
                self._evict_task(evict_id)

        # Return selected task as CurriculumTask
        task_sample = self.task_pool.get_task(selected_task_id)
        if task_sample is None:
            raise ValueError(f"Selected task ID {selected_task_id} not found in task pool")

        # Get or create the CurriculumTask object and ensure it's in _tasks
        if selected_task_id in self._tasks:
            curriculum_task = self._tasks[selected_task_id]
        else:
            curriculum_task = self._task_sample_to_curriculum_task(task_sample)
            self._tasks[selected_task_id] = curriculum_task

        # Update scheduling counter for backwards compatibility
        curriculum_task._num_scheduled += 1

        return curriculum_task

    def _sample_by_scores(self, scores: Dict[int, float]) -> int:
        """Sample task ID based on scores"""
        available_task_ids = self.task_pool.get_all_task_ids()
        if not scores or not available_task_ids:
            raise ValueError("No valid tasks available for sampling")

        task_ids = list(scores.keys())
        weights = [scores[tid] for tid in task_ids]
        # Ensure we have positive weights
        if sum(weights) <= 0:
            weights = [1.0] * len(task_ids)
        return self._rng.choices(task_ids, weights=weights)[0]

    def _evict_task(self, task_id: int) -> None:
        """Evict a task and notify algorithm"""
        # Remove from backwards compatibility _tasks dict first
        if task_id in self._tasks:
            del self._tasks[task_id]

        self.algorithm.on_task_evicted(task_id)
        self.task_pool.evict_task(task_id)  # This creates a new task automatically
        if self.task_tracker:
            self.task_tracker.remove_task(task_id)

        # Update eviction counter and backwards compatibility
        self._num_evicted += 1
        self._num_created += 1  # New task was created to replace evicted one

        # Add the new task to backwards compatibility _tasks dict
        # Find the newest task (highest task_id) and add it
        all_tasks = self.task_pool.get_all_tasks()
        if all_tasks:
            newest_task = max(all_tasks, key=lambda t: t.task_id)
            if newest_task.task_id not in self._tasks:
                curriculum_task = self._task_sample_to_curriculum_task(newest_task)
                self._tasks[newest_task.task_id] = curriculum_task

    def _task_sample_to_curriculum_task(self, task_sample: TaskSample) -> CurriculumTask:
        """Convert TaskSample to CurriculumTask for external API"""
        # For now, create a simple env_cfg from the stored information
        # In a full implementation, this would regenerate the task from env_class and seed
        env_cfg = {"task_id": task_sample.task_id, "seed": task_sample.seed}
        return CurriculumTask(task_sample.task_id, env_cfg, task_sample.bucket_values)

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update performance across all components"""
        self.algorithm.update_task_performance(task_id, score)
        self.task_pool.update_task_performance(task_id, score)

        if self.task_tracker:
            task = self.task_pool.get_task(task_id)
            bucket_values = task.bucket_values if task else {}
            self.task_tracker.update_task_performance(task_id, score, bucket_values)

    def _initialize_tasks_dict(self):
        """Initialize backwards compatibility _tasks dict"""
        for task_sample in self.task_pool.get_all_tasks():
            curriculum_task = self._task_sample_to_curriculum_task(task_sample)
            self._tasks[task_sample.task_id] = curriculum_task

    def _task_sample_to_curriculum_task(self, task_sample: TaskSample) -> CurriculumTask:
        """Convert TaskSample to CurriculumTask for backwards compatibility"""
        env_cfg = self._task_generator.get_task(task_sample.task_id)
        return CurriculumTask(task_sample.task_id, env_cfg)

    def stats(self, prefix: str = "") -> Dict[str, float]:
        """Get comprehensive curriculum statistics"""
        stats = {}

        # Algorithm stats
        algo_stats = self.algorithm.stats(f"{prefix}algorithm/")
        stats.update(algo_stats)

        # Task pool stats
        stats[f"{prefix}pool/num_tasks"] = float(len(self.task_pool.get_all_tasks()))
        stats[f"{prefix}pool/capacity"] = float(self.task_pool.capacity)

        # Backwards compatibility stats
        stats[f"{prefix}num_created"] = float(self._num_created)
        stats[f"{prefix}num_evicted"] = float(self._num_evicted)
        stats[f"{prefix}num_active_tasks"] = float(len(self.task_pool.get_all_tasks()))
        stats[f"{prefix}num_completed"] = float(sum(task._num_completions for task in self._tasks.values()))
        stats[f"{prefix}num_scheduled"] = float(sum(task._num_scheduled for task in self._tasks.values()))

        # Bucket tracking stats
        if self.task_tracker:
            tracker_stats = self.task_tracker.get_global_stats()
            for key, value in tracker_stats.items():
                stats[f"{prefix}tracker/{key}"] = value

        return stats
