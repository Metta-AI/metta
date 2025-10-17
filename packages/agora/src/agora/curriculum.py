"""Core curriculum implementations and utilities."""

from __future__ import annotations

import abc
import logging
import random
from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar, Generic

from pydantic import BaseModel, ConfigDict, Field

from agora.config import TConfig
from agora.tracking.stats import SliceAnalyzer, StatsLogger

if TYPE_CHECKING:
    from agora.generators.base import TaskGenerator

logger = logging.getLogger(__name__)


class CurriculumTask(Generic[TConfig]):
    """A task instance with a task_id and task configuration.

    Generic over TConfig which should be any type implementing TaskConfig protocol.
    """

    def __init__(self, task_id: int, task_config: TConfig, slice_values: dict[str, Any] | None = None):
        """Initialize curriculum task.

        Args:
            task_id: Unique identifier for this task
            task_config: Task configuration (any type implementing TaskConfig protocol)
            slice_values: Optional slice/bucket values used to generate this task
        """
        self._task_id = task_id
        self._task_config = task_config
        self._slice_values = slice_values or {}
        self._num_completions = 0
        self._total_score = 0.0
        self._mean_score = 0.0
        self._num_scheduled = 0
        # Extract label from config for per-label logging
        self._label = getattr(task_config, "label", "unknown")

    def complete(self, score: float) -> None:
        """Complete the task with a score.

        Args:
            score: Performance score for this task completion
        """
        self._num_completions += 1
        self._total_score += score
        self._mean_score = self._total_score / self._num_completions

    def get_env_cfg(self) -> TConfig:
        """Get the task configuration.

        Returns:
            Task configuration of type TConfig
        """
        return self._task_config

    def get_slice_values(self) -> dict[str, Any]:
        """Get the slice values that were used to generate this task.

        Returns:
            Dictionary of slice/bucket values
        """
        return self._slice_values

    def get_bucket_values(self) -> dict[str, Any]:
        """Get the slice values (backward compatibility alias).

        Returns:
            Dictionary of slice/bucket values
        """
        return self._slice_values

    def get_label(self) -> str:
        """Get the task label for per-label metrics.

        Returns:
            Task label string
        """
        return self._label

    # Backward compatibility: expose internal attributes as properties
    @property
    def task_id(self) -> int:
        """Task ID (backward compatibility)."""
        return self._task_id

    @property
    def env_cfg(self) -> TConfig:
        """Environment config (backward compatibility)."""
        return self._task_config

    @property
    def _env_cfg(self) -> TConfig:
        """Direct access to env_cfg (backward compatibility for tests)."""
        return self._task_config

    @property
    def slice_values(self) -> dict[str, Any]:
        """Slice values (backward compatibility)."""
        return self._slice_values

    @property
    def num_completions(self) -> int:
        """Number of completions (backward compatibility)."""
        return self._num_completions

    @property
    def mean_score(self) -> float:
        """Mean score (backward compatibility)."""
        return self._mean_score


class CurriculumAlgorithmConfig(BaseModel, ABC):
    """Hyperparameters for the CurriculumAlgorithm."""

    type: str = Field(description="Type of algorithm hyperparameters")
    initial_weights: list[float] | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    @abc.abstractmethod
    def algorithm_type(self) -> str:
        """Return the algorithm type string used in configs."""
        ...

    def create(self, num_tasks: int) -> CurriculumAlgorithm:
        """Create the curriculum algorithm with these hyperparameters.

        Args:
            num_tasks: Number of tasks the algorithm will manage

        Returns:
            Configured curriculum algorithm instance
        """
        # Allow subclasses to override this
        # Default to DiscreteRandomCurriculum
        return DiscreteRandomCurriculum(num_tasks, self)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Dump model to dict, ensuring we return the config object not a dict."""
        data = super().model_dump(**kwargs)
        # Mark this as a config object, not just a dict
        return data


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
    def score_tasks(self, task_ids: list[int]) -> dict[int, float]:
        """Score tasks for selection purposes. Higher scores = more likely to be selected."""
        ...

    @abc.abstractmethod
    def recommend_eviction(self, task_ids: list[int]) -> int | None:
        """Recommend which task to evict. Return None for random selection."""
        ...

    @abc.abstractmethod
    def on_task_evicted(self, task_id: int) -> None:
        """Notification that a task has been evicted from the pool."""
        ...

    @abc.abstractmethod
    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance. Override in subclasses that track performance."""
        ...

    def get_state(self) -> dict[str, Any]:
        """Get algorithm state for checkpointing. Override in subclasses that have state."""
        return {"type": self.hypers.algorithm_type()}

    def load_state(self, state: dict[str, Any]) -> None:
        """Load algorithm state from checkpoint. Override in subclasses that have state."""
        pass

    def on_task_created(self, task: CurriculumTask[Any]) -> None:
        """Notification that a new task has been created. Override if needed."""
        pass

    def set_curriculum_reference(self, curriculum: Curriculum[Any]) -> None:
        """Set reference to curriculum for stats updates. Override if needed."""
        pass

    def should_evict_task(self, task_id: int, min_presentations: int) -> bool:
        """Check if a task should be evicted based on algorithm-specific criteria.

        Default implementation returns False (no eviction). Subclasses should override
        to implement their own eviction criteria.

        Args:
            task_id: The task to check
            min_presentations: Minimum number of task presentations before eviction
                              (should be passed from CurriculumConfig.min_presentations_for_eviction)

        Returns:
            True if task should be evicted
        """
        return False

    def __init__(
        self,
        num_tasks: int,
        hypers: CurriculumAlgorithmConfig | None = None,
        initialize_weights: bool = True,
    ):
        """Initialize curriculum algorithm.

        Args:
            num_tasks: Number of tasks to manage
            hypers: Algorithm hyperparameters
            initialize_weights: Whether to initialize weights

        Raises:
            ValueError: If num_tasks is not positive
        """
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

    def get_base_stats(self) -> dict[str, float]:
        """Get basic statistics that all algorithms must provide."""
        return {"num_tasks": self.num_tasks, **self.slice_analyzer.get_base_stats()}

    def get_detailed_stats(self) -> dict[str, float]:
        """Get detailed stats including expensive slice analysis."""
        return self.slice_analyzer.get_detailed_stats()

    def stats(self, prefix: str = "") -> dict[str, float]:
        """Return statistics for logging purposes. Add `prefix` to all keys."""
        # Use the StatsLogger implementation
        return super().stats(prefix)


class DiscreteRandomConfig(CurriculumAlgorithmConfig):
    """Hyperparameters for DiscreteRandomCurriculum."""

    type: str = "discrete_random"

    def algorithm_type(self) -> str:
        """Return algorithm type string."""
        return "discrete_random"


class DiscreteRandomCurriculum(CurriculumAlgorithm):
    """Curriculum algorithm that samples from a discrete distribution of weights.

    A named class for the simplest case where weights don't change based on
    task performance.
    """

    def score_tasks(self, task_ids: list[int]) -> dict[int, float]:
        """All tasks have equal score for random selection."""
        return {task_id: 1.0 for task_id in task_ids}

    def recommend_eviction(self, task_ids: list[int]) -> int | None:
        """No preference for eviction - let Curriculum choose randomly."""
        return None

    def on_task_evicted(self, task_id: int) -> None:
        """No action needed for random curriculum."""
        pass

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance - no-op for discrete random curriculum."""
        pass


class CurriculumConfig(BaseModel, Generic[TConfig]):
    """Base configuration for Curriculum.

    Generic over TConfig which should be any type implementing TaskConfig protocol.
    """

    task_generator: Any = Field(description="TaskGenerator configuration")  # TaskGeneratorConfig[TConfig]
    max_task_id: int = Field(default=1000000, gt=0, description="Maximum task ID to generate")
    num_active_tasks: int = Field(default=1000, gt=0, description="Number of active tasks to maintain")

    # Curriculum behavior options
    seed: int = Field(default=0, description="Random seed for curriculum task generation")
    defer_init: bool = Field(default=False, description="Defer task pool initialization (used for checkpoint loading)")
    min_presentations_for_eviction: int = Field(
        default=5, gt=0, description="Minimum task presentations before eviction"
    )

    algorithm_config: Any = Field(default=None, description="Curriculum algorithm hyperparameters")

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate configuration after initialization."""
        super().model_post_init(__context)

        # Convert algorithm_config from dict to object if needed
        if self.algorithm_config is not None and isinstance(self.algorithm_config, dict):
            from agora.algorithms.learning_progress import LearningProgressConfig

            algo_dict = self.algorithm_config.copy()
            algo_type = algo_dict.pop("type", "learning_progress")
            if algo_type == "learning_progress":
                self.algorithm_config = LearningProgressConfig(**algo_dict)

        # Sync num_active_tasks from algorithm_config if available
        if self.algorithm_config and hasattr(self.algorithm_config, "num_active_tasks"):
            self.num_active_tasks = self.algorithm_config.num_active_tasks

        if self.num_active_tasks > self.max_task_id:
            raise ValueError(
                f"num_active_tasks ({self.num_active_tasks}) cannot exceed max_task_id ({self.max_task_id})"
            )

    def make(self) -> Curriculum[TConfig]:
        """Create a Curriculum from this configuration."""
        return Curriculum(self)

    @classmethod
    def from_mg(cls, mg_config: Any) -> CurriculumConfig[Any]:
        """Create a CurriculumConfig from a MettaGridConfig.

        Backward compatibility helper for MettaGrid-specific usage.

        Args:
            mg_config: MettaGridConfig instance

        Returns:
            CurriculumConfig with SingleTaskGenerator
        """
        # Import here to avoid circular dependency
        from agora.generators.single import SingleTaskGenerator

        return cls(
            task_generator=SingleTaskGenerator.Config(env=mg_config),
        )


class Curriculum(StatsLogger, Generic[TConfig]):
    """Base curriculum class that uses TaskGenerator to generate task configs and returns Tasks.

    Curriculum takes a CurriculumConfig, and supports get_task(). It uses the task generator
    to generate the task configuration and then returns a CurriculumTask. It can optionally use a
    CurriculumAlgorithm for intelligent task selection.

    Generic over TConfig which should be any type implementing TaskConfig protocol.
    Inherits from StatsLogger to provide unified statistics interface.
    """

    def __init__(self, config: CurriculumConfig[TConfig]):
        """Initialize curriculum.

        Args:
            config: Curriculum configuration
        """
        # Initialize StatsLogger (algorithm handles detailed stats)
        StatsLogger.__init__(self, enable_detailed_logging=False)

        self._config = config
        self._task_generator: TaskGenerator[TConfig] = config.task_generator.create()
        self._rng = random.Random(config.seed)
        self._tasks: dict[int, CurriculumTask[TConfig]] = {}
        self._task_ids: set[int] = set()
        self._num_created = 0
        self._num_evicted = 0

        self._algorithm: CurriculumAlgorithm | None = None
        if config.algorithm_config is not None:
            # Handle both dict and object forms
            if isinstance(config.algorithm_config, dict):
                # Convert dict to proper config object
                from agora.algorithms.learning_progress import LearningProgressConfig

                algo_dict = config.algorithm_config.copy()
                algo_type = algo_dict.pop("type", "learning_progress")
                if algo_type == "learning_progress":
                    config.algorithm_config = LearningProgressConfig(**algo_dict)

            # Always use algorithm_config.num_active_tasks if available (source of truth for algorithm-based curricula)
            num_tasks = (
                config.algorithm_config.num_active_tasks
                if hasattr(config.algorithm_config, "num_active_tasks")
                else config.num_active_tasks
            )
            self._algorithm = config.algorithm_config.create(num_tasks)
            # Pass curriculum reference to algorithm for stats updates
            if hasattr(self._algorithm, "set_curriculum_reference"):
                self._algorithm.set_curriculum_reference(self)

        # Initialize task pool at capacity unless deferred (e.g., for checkpoint loading)
        if not config.defer_init:
            self._initialize_at_capacity()

    @property
    def _num_active_tasks(self) -> int:
        """Get the effective number of active tasks.

        Handles three scenarios:
        1. No algorithm_config: use config.num_active_tasks
        2. Algorithm_config present and in sync with config: use algorithm_config (allows CLI overrides)
        3. Algorithm_config present but out of sync: use config (manual override after creation)
        """
        if self._config.algorithm_config is None or not hasattr(self._config.algorithm_config, "num_active_tasks"):
            return self._config.num_active_tasks

        # If they match, they're in sync - use algorithm_config (allows CLI overrides to work)
        if self._config.algorithm_config.num_active_tasks == self._config.num_active_tasks:
            return self._config.algorithm_config.num_active_tasks

        # If they differ, config was manually overridden - prefer config
        return self._config.num_active_tasks

    def get_task(self) -> CurriculumTask[TConfig]:
        """Sample a task from the population.

        Returns:
            A curriculum task with configuration
        """
        # Curriculum always manages the task pool - no delegation
        if len(self._tasks) < self._num_active_tasks:
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
        while len(self._tasks) < self._num_active_tasks:
            self._create_task()

    def _evict_specific_task(self, task_id: int) -> None:
        """Evict a specific task by ID.

        Args:
            task_id: ID of task to evict
        """
        if task_id not in self._tasks:
            return

        # Notify algorithm of eviction
        if self._algorithm is not None:
            self._algorithm.on_task_evicted(task_id)

        self._task_ids.remove(task_id)
        self._tasks.pop(task_id)
        self._num_evicted += 1

    def _choose_task(self) -> CurriculumTask[TConfig]:
        """Choose a task from the population using algorithm guidance.

        Returns:
            A curriculum task
        """
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

    def _create_task(self) -> CurriculumTask[TConfig]:
        """Create a new task.

        Returns:
            A new curriculum task
        """
        task_id = self._rng.randint(0, self._config.max_task_id)
        while task_id in self._task_ids:
            task_id = self._rng.randint(0, self._config.max_task_id)
        self._task_ids.add(task_id)
        task_config = self._task_generator.get_task(task_id)

        # Extract bucket values if available
        bucket_values = {}
        if hasattr(self._task_generator, "_last_bucket_values"):
            bucket_values = self._task_generator._last_bucket_values.copy()  # type: ignore[attr-defined]

        task = CurriculumTask(task_id, task_config, bucket_values)
        self._tasks[task_id] = task
        self._num_created += 1

        # Notify algorithm of new task
        if self._algorithm is not None and hasattr(self._algorithm, "on_task_created"):
            self._algorithm.on_task_created(task)

        return task

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update the curriculum algorithm with task performance.

        Args:
            task_id: Task identifier
            score: Performance score
        """
        if self._algorithm is not None:
            self._algorithm.update_task_performance(task_id, score)

        # Invalidate stats cache since task performance affects curriculum stats
        self.invalidate_cache()

    def get_base_stats(self) -> dict[str, float]:
        """Get basic curriculum statistics."""
        base_stats: dict[str, float] = {
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

    def stats(self) -> dict[str, float]:
        """Return curriculum statistics for logging purposes."""
        # Use the StatsLogger implementation
        return super().stats()

    def get_state(self) -> dict[str, Any]:
        """Get curriculum state for checkpointing."""
        state: dict[str, Any] = {
            "config": self._config.model_dump(),  # Save config for validation
            "seed": self._rng.getstate(),
            "num_created": self._num_created,
            "num_evicted": self._num_evicted,
            "tasks": {},
        }

        # Serialize task data (without task_config to save space)
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

    def load_state(self, state: dict[str, Any]) -> None:
        """Load curriculum state from checkpoint.

        Args:
            state: Checkpoint state dictionary
        """
        # Validate config matches
        if state["config"] != self._config.model_dump():
            logger.warning("Curriculum config mismatch during restore")

        # Restore counters first
        self._num_created = state["num_created"]
        self._num_evicted = state["num_evicted"]

        # Restore random state before any RNG operations
        self._rng.setstate(state["seed"])

        # Clear existing tasks (no need to notify algorithm - we're doing full restore)
        self._tasks.clear()
        self._task_ids.clear()

        # Restore algorithm state BEFORE recreating tasks
        # Algorithm's load_state will handle clearing and restoring its internal state atomically
        if self._algorithm is not None and "algorithm_state" in state:
            self._algorithm.load_state(state["algorithm_state"])

        # Restore tasks
        for task_id_str, task_data in state["tasks"].items():
            # Recreate task_config using task_id
            task_id = int(task_id_str)
            task_config = self._task_generator.get_task(task_id)
            task = CurriculumTask(task_id, task_config, task_data["slice_values"])
            task._num_completions = task_data["num_completions"]
            task._total_score = task_data["total_score"]
            task._mean_score = task_data["mean_score"]
            task._num_scheduled = task_data["num_scheduled"]

            self._tasks[task_id] = task
            self._task_ids.add(task_id)

        # NOTE: We don't call on_task_created() here because:
        # 1. Algorithm state (including task_tracker) is already restored above via load_state()
        # 2. Calling it would re-initialize tracking with default values
        # 3. slice_analyzer state is not checkpointed, so it will rebuild naturally


# Note: LearningProgressConfig is a forward reference to avoid circular imports
# It will be resolved when the model is used


__all__ = [
    "CurriculumTask",
    "CurriculumAlgorithmConfig",
    "CurriculumAlgorithm",
    "DiscreteRandomConfig",
    "DiscreteRandomCurriculum",
    "CurriculumConfig",
    "Curriculum",
]
