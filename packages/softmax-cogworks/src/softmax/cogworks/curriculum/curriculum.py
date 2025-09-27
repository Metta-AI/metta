"""Core curriculum implementations and utilities."""

from __future__ import annotations

import abc
import logging
import random
from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union

if TYPE_CHECKING:
    from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig

from pydantic import ConfigDict, Field

from metta.cogworks.curriculum.stats import SliceAnalyzer, StatsLogger
from metta.cogworks.curriculum.task_generator import AnyTaskGeneratorConfig, SingleTaskGenerator
from mettagrid.config import Config
from mettagrid.config.mettagrid_config import MettaGridConfig

logger = logging.getLogger(__name__)


def get_algorithm_hypers_discriminator(v):
    """Discriminator function for algorithm hypers types."""
    if isinstance(v, dict) and "type" in v:
        return v["type"]
    return None


class CurriculumTask:
    """A task instance with a task_id and env_cfg."""

    def __init__(self, task_id: int, env_cfg, slice_values: Optional[Dict[str, Any]] = None):
        self._task_id = task_id
        self._env_cfg = env_cfg
        self._slice_values = slice_values or {}
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

    def get_slice_values(self):
        """Get the slice values that were used to generate this task."""
        return self._slice_values

    def get_bucket_values(self):
        """Get the slice values (backward compatibility alias)."""
        return self._slice_values


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


class CurriculumAlgorithm(StatsLogger, ABC):
    """
    Curriculum algorithms are responsible for:
    1. Scoring tasks based on their learning progress or other metrics
    2. Recommending which tasks to evict when the pool is full
    3. Tracking task performance for algorithm-specific purposes
    4. Providing feedback to Curriculum for task selection

    The Curriculum maintains the task pool and lifecycle, while algorithms provide guidance.
    Inherits from StatsLogger to provide unified statistics interface.
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

    @abc.abstractmethod
    def on_task_evicted(self, task_id: int) -> None:
        """Notification that a task has been evicted from the pool."""
        pass

    @abc.abstractmethod
    def update_task_performance(self, task_id: int, score: float):
        """Update task performance. Override in subclasses that track performance."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get algorithm state for checkpointing. Override in subclasses that have state."""
        return {"type": self.hypers.algorithm_type()}

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load algorithm state from checkpoint. Override in subclasses that have state."""
        pass

    def on_task_created(self, task: "CurriculumTask") -> None:
        """Notification that a new task has been created. Override if needed."""
        pass

    def set_curriculum_reference(self, curriculum: "Curriculum") -> None:
        """Set reference to curriculum for stats updates. Override if needed."""
        pass

    def should_evict_task(self, task_id: int, min_presentations: int = 5) -> bool:
        """Check if a task should be evicted based on algorithm-specific criteria.

        Default implementation returns False (no eviction). Subclasses should override
        to implement their own eviction criteria.

        Args:
            task_id: The task to check
            min_presentations: Minimum number of task presentations before eviction

        Returns:
            True if task should be evicted
        """
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

        # Initialize stats logging
        enable_detailed = getattr(hypers, "enable_detailed_slice_logging", False)
        StatsLogger.__init__(self, enable_detailed_logging=enable_detailed)

        # All algorithms get slice analysis capability
        max_slice_axes = getattr(hypers, "max_slice_axes", 3)
        self.slice_analyzer = SliceAnalyzer(max_slice_axes=max_slice_axes, enable_detailed_logging=enable_detailed)

    def get_base_stats(self) -> Dict[str, float]:
        """Get basic statistics that all algorithms must provide."""
        return {"num_tasks": self.num_tasks, **self.slice_analyzer.get_base_stats()}

    def get_detailed_stats(self) -> Dict[str, float]:
        """Get detailed stats including expensive slice analysis."""
        return self.slice_analyzer.get_detailed_stats()

    def stats(self, prefix: str = "") -> dict[str, float]:
        """Return statistics for logging purposes. Add `prefix` to all keys."""
        # Use the StatsLogger implementation
        return super().stats(prefix)

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

    def on_task_evicted(self, task_id: int) -> None:
        """No action needed for random curriculum."""
        pass

    def update_task_performance(self, task_id: int, score: float):
        """Update task performance - no-op for discrete random curriculum."""
        pass


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
            task_generator=SingleTaskGenerator.Config(env=mg_config),
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


class Curriculum(StatsLogger):
    """Base curriculum class that uses TaskGenerator to generate EnvConfigs and returns Tasks.

    Curriculum takes a CurriculumConfig, and supports get_task(). It uses the task generator
    to generate the EnvConfig and then returns a Task(env_cfg). It can optionally use a
    CurriculumAlgorithm for intelligent task selection.

    Inherits from StatsLogger to provide unified statistics interface.
    """

    def __init__(self, config: CurriculumConfig, seed: int = 0):
        # Initialize StatsLogger (algorithm handles detailed stats)
        StatsLogger.__init__(self, enable_detailed_logging=False)

        self._config = config
        self._task_generator = config.task_generator.create()
        self._rng = random.Random(seed)
        self._tasks: dict[int, CurriculumTask] = {}
        self._task_ids: set[int] = set()
        self._num_created = 0
        self._num_evicted = 0

        self._algorithm: Optional[CurriculumAlgorithm] = None
        if config.algorithm_config is not None:
            self._algorithm = config.algorithm_config.create(config.num_active_tasks)
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
            # At capacity - check if any task meets eviction criteria first
            task = None
            if self._algorithm is not None:
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

            # If no eviction happened, choose from existing tasks
            if task is None:
                task = self._choose_task()

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
        if self._algorithm is not None:
            self._algorithm.on_task_evicted(task_id)

        self._task_ids.remove(task_id)
        self._tasks.pop(task_id)
        self._num_evicted += 1

    def _choose_task(self) -> CurriculumTask:
        """Choose a task from the population using algorithm guidance."""
        if self._algorithm is not None:
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

        # Fallback to random selection
        return self._tasks[self._rng.choice(list(self._tasks.keys()))]

    def _create_task(self) -> CurriculumTask:
        """Create a new task."""
        task_id = self._rng.randint(0, self._config.max_task_id)
        while task_id in self._task_ids:
            task_id = self._rng.randint(0, self._config.max_task_id)
        self._task_ids.add(task_id)
        env_cfg = self._task_generator.get_task(task_id)

        # Extract bucket values if available
        bucket_values = {}
        if hasattr(self._task_generator, "_last_bucket_values"):
            bucket_values = self._task_generator._last_bucket_values.copy()

        task = CurriculumTask(task_id, env_cfg, bucket_values)
        self._tasks[task_id] = task
        self._num_created += 1

        # Notify algorithm of new task
        if self._algorithm is not None and hasattr(self._algorithm, "on_task_created"):
            self._algorithm.on_task_created(task)

        return task

    def update_task_performance(self, task_id: int, score: float):
        """Update the curriculum algorithm with task performance."""
        if self._algorithm is not None:
            self._algorithm.update_task_performance(task_id, score)

        # Invalidate stats cache since task performance affects curriculum stats
        self.invalidate_cache()

    def get_base_stats(self) -> Dict[str, float]:
        """Get basic curriculum statistics."""
        base_stats: Dict[str, float] = {
            "num_created": float(self._num_created),
            "num_evicted": float(self._num_evicted),
            "num_completed": float(sum(task._num_completions for task in self._tasks.values())),
            "num_scheduled": float(sum(task._num_scheduled for task in self._tasks.values())),
            "num_active_tasks": float(len(self._tasks)),
        }

        # Include algorithm stats if available
        if self._algorithm is not None:
            algorithm_stats = self._algorithm.stats("algorithm/")
            base_stats.update(algorithm_stats)

        return base_stats

    def stats(self) -> dict:
        """Return curriculum statistics for logging purposes."""
        # Use the StatsLogger implementation
        return super().stats()

    def get_state(self) -> Dict[str, Any]:
        """Get curriculum state for checkpointing."""
        state = {
            "config": self._config.model_dump(),  # Save config for validation
            "seed": self._rng.getstate(),
            "num_created": self._num_created,
            "num_evicted": self._num_evicted,
            "tasks": {},
        }

        # Serialize task data (without env_cfg to save space)
        for task_id, task in self._tasks.items():
            state["tasks"][task_id] = {
                "num_completions": task._num_completions,
                "total_score": task._total_score,
                "mean_score": task._mean_score,
                "num_scheduled": task._num_scheduled,
                "slice_values": task._slice_values,
            }

        # Save algorithm state if present
        if self._algorithm is not None:
            state["algorithm_state"] = self._algorithm.get_state()

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load curriculum state from checkpoint."""
        # Validate config matches
        if state["config"] != self._config.model_dump():
            logger.warning("Curriculum config mismatch during restore")

        # Restore counters
        self._num_created = state["num_created"]
        self._num_evicted = state["num_evicted"]

        # Restore random state
        self._rng.setstate(state["seed"])

        # Clear existing tasks
        self._tasks.clear()
        self._task_ids.clear()

        # Restore tasks
        for task_id_str, task_data in state["tasks"].items():
            # Recreate env_cfg using task_id
            task_id = int(task_id_str)
            env_cfg = self._task_generator.get_task(task_id)
            task = CurriculumTask(task_id, env_cfg, task_data["slice_values"])
            task._num_completions = task_data["num_completions"]
            task._total_score = task_data["total_score"]
            task._mean_score = task_data["mean_score"]
            task._num_scheduled = task_data["num_scheduled"]

            self._tasks[task_id] = task
            self._task_ids.add(task_id)

        # Restore algorithm state
        if self._algorithm is not None and "algorithm_state" in state:
            self._algorithm.load_state(state["algorithm_state"])


# Import concrete config classes at the end to avoid circular imports
# ruff: noqa: E402
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig

# Rebuild the model to resolve forward references
CurriculumConfig.model_rebuild()
