"""Core curriculum implementations and utilities."""

from __future__ import annotations

import abc
import logging
import multiprocessing
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

    # Two-pool curriculum parameters
    # Pool capacities auto-calculate if not provided (~4% explore, ~96% exploit)
    explore_pool_capacity: Optional[int] = Field(
        default=None, gt=0, description="Maximum number of tasks in explore pool (defaults to ~4% of num_active_tasks)"
    )
    exploit_pool_capacity: Optional[int] = Field(
        default=None, gt=0, description="Maximum number of tasks in exploit pool (defaults to ~96% of num_active_tasks)"
    )
    promotion_threshold: int = Field(default=10, gt=0, description="Number of presentations before promotion attempt")
    min_explore_rate: float = Field(default=0.01, gt=0, lt=1, description="Minimum guaranteed exploration probability")
    alpha: float = Field(default=0.1, gt=0, lt=1, description="Smoothing factor for EMA of acceptance rate")

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

        # Two-pool system requires at least 2 tasks - auto-adjust if needed
        if self.num_active_tasks < 2:
            logger.warning(
                f"Two-pool curriculum requires num_active_tasks >= 2, got {self.num_active_tasks}. "
                f"Auto-adjusting to 2 (min: 1 explore + 1 exploit)."
            )
            self.num_active_tasks = 2
            # Also adjust max_task_id if needed
            if self.max_task_id < self.num_active_tasks:
                self.max_task_id = self.num_active_tasks

        if self.num_active_tasks > self.max_task_id:
            raise ValueError(
                f"num_active_tasks ({self.num_active_tasks}) cannot exceed max_task_id ({self.max_task_id})"
            )

        # Auto-calculate pool capacities if not provided
        # Use ~4% for explore pool (min 1), rest for exploit pool
        if self.explore_pool_capacity is None and self.exploit_pool_capacity is None:
            # Neither provided - auto-calculate both
            # For small values, use 1 explore task; for larger values use ~4%
            self.explore_pool_capacity = max(1, min(int(self.num_active_tasks * 0.04), self.num_active_tasks - 1))
            self.exploit_pool_capacity = self.num_active_tasks - self.explore_pool_capacity
        elif self.explore_pool_capacity is None:
            # Only exploit provided - calculate explore
            self.explore_pool_capacity = self.num_active_tasks - self.exploit_pool_capacity
        elif self.exploit_pool_capacity is None:
            # Only explore provided - calculate exploit
            self.exploit_pool_capacity = self.num_active_tasks - self.explore_pool_capacity

        # Validate two-pool configuration
        if self.explore_pool_capacity + self.exploit_pool_capacity != self.num_active_tasks:
            raise ValueError(
                f"explore_pool_capacity ({self.explore_pool_capacity}) + "
                f"exploit_pool_capacity ({self.exploit_pool_capacity}) must equal "
                f"num_active_tasks ({self.num_active_tasks})"
            )

        # Ensure both pools have at least 1 task
        if self.explore_pool_capacity < 1:
            raise ValueError(f"explore_pool_capacity must be at least 1, got {self.explore_pool_capacity}")
        if self.exploit_pool_capacity < 1:
            raise ValueError(f"exploit_pool_capacity must be at least 1, got {self.exploit_pool_capacity}")

    def make(self) -> "Curriculum":
        """Create a Curriculum from this configuration."""
        return Curriculum(self)


class Curriculum(StatsLogger):
    """Two-pool curriculum class with Explore/Exploit strategy.

    Maintains two separate task pools:
    - Explore pool: Small pool for newly created tasks
    - Exploit pool: Larger pool for tasks with proven learning value

    Tasks are promoted from explore to exploit based on Learning Progress Score (LPS).
    The promotion acceptance rate (P_accept) is tracked via EMA to adaptively balance
    exploration and exploitation.

    Inherits from StatsLogger to provide unified statistics interface.
    """

    # Class-level atomic counter for process-safe task ID generation
    _task_id_counter: Optional[multiprocessing.Value] = None
    _counter_lock: Optional[multiprocessing.Lock] = None

    @classmethod
    def _init_shared_counter(cls):
        """Initialize shared counter for multi-process task ID generation."""
        if cls._task_id_counter is None:
            cls._task_id_counter = multiprocessing.Value("i", 0)
            cls._counter_lock = multiprocessing.Lock()

    def __init__(self, config: CurriculumConfig, seed: int = 0):
        # Initialize StatsLogger (algorithm handles detailed stats)
        StatsLogger.__init__(self, enable_detailed_logging=False)

        self._config = config
        self._task_generator = config.task_generator.create()
        self._rng = random.Random(seed)
        self._num_created = 0
        self._num_evicted = 0

        # Initialize shared counter for process-safe task ID generation
        self._init_shared_counter()

        # Two separate pools for explore/exploit
        self._explore_pool: dict[int, CurriculumTask] = {}
        self._exploit_pool: dict[int, CurriculumTask] = {}
        self._task_ids: set[int] = set()

        # Promotion acceptance rate (running EMA)
        self._P_accept: float = 0.5

        # Track number of promotions attempted and accepted
        self._num_promotions_attempted = 0
        self._num_promotions_accepted = 0

        # For backward compatibility with tests, create _tasks property
        # that merges both pools
        self._tasks_property_warning_shown = False

        self._algorithm: Optional[CurriculumAlgorithm] = None
        if config.algorithm_config is not None:
            self._algorithm = config.algorithm_config.create(config.num_active_tasks)
            # Pass curriculum reference to algorithm for stats updates
            if hasattr(self._algorithm, "set_curriculum_reference"):
                self._algorithm.set_curriculum_reference(self)

        # Always initialize task pools at capacity
        self._initialize_at_capacity()

    @property
    def _tasks(self) -> dict[int, CurriculumTask]:
        """Backward compatibility property that merges both pools.

        This property is provided for backward compatibility with existing tests
        and code that expects a single _tasks dict. It returns a merged view of
        both explore and exploit pools.
        """
        if not self._tasks_property_warning_shown:
            logger.warning(
                "Accessing _tasks property for backward compatibility. "
                "Please update code to use _explore_pool and _exploit_pool directly."
            )
            self._tasks_property_warning_shown = True
        return {**self._explore_pool, **self._exploit_pool}

    def get_task(self) -> CurriculumTask:
        """Two-pool mode task selection."""
        # Fill explore pool if not at capacity
        if len(self._explore_pool) < self._config.explore_pool_capacity:
            task = self._create_task(pool="explore")
            task._num_scheduled += 1
            return task

        # Both pools should be at capacity at this point (or exploit pool filling)
        if len(self._exploit_pool) < self._config.exploit_pool_capacity:
            # Still filling exploit pool - create new tasks for it
            task = self._create_task(pool="exploit")
            task._num_scheduled += 1
            return task

        # Both pools at capacity - probabilistically choose pool
        effective_explore_rate = max(self._P_accept, self._config.min_explore_rate)

        # Defensive check: ensure pools are not empty before sampling
        if len(self._explore_pool) == 0 and len(self._exploit_pool) == 0:
            logger.error("Both pools are empty! Creating emergency task.")
            task = self._create_task(pool="explore")
            task._num_scheduled += 1
            return task

        # If one pool is empty, use the other
        if len(self._explore_pool) == 0:
            logger.warning("Explore pool is empty, sampling from exploit pool only")
            task = self._choose_task_from_pool(self._exploit_pool)
        elif len(self._exploit_pool) == 0:
            logger.warning("Exploit pool is empty, sampling from explore pool only")
            task = self._choose_task_from_pool(self._explore_pool)
        elif self._rng.random() < effective_explore_rate:
            # Sample from explore pool
            task = self._choose_task_from_pool(self._explore_pool)
        else:
            # Sample from exploit pool
            task = self._choose_task_from_pool(self._exploit_pool)

        task._num_scheduled += 1
        return task

    def _initialize_at_capacity(self) -> None:
        """Initialize both task pools to full capacity."""
        # Fill explore pool first
        while len(self._explore_pool) < self._config.explore_pool_capacity:
            self._create_task(pool="explore")
        # Then fill exploit pool
        while len(self._exploit_pool) < self._config.exploit_pool_capacity:
            self._create_task(pool="exploit")

    def _choose_task_from_pool(self, pool: dict[int, CurriculumTask]) -> CurriculumTask:
        """Choose a task from a specific pool using algorithm guidance."""
        if not pool:
            logger.error(
                f"Pool is empty! Explore pool size: {len(self._explore_pool)}, "
                f"Exploit pool size: {len(self._exploit_pool)}. This indicates a bug in task management."
            )
            raise ValueError("Cannot choose from empty pool")

        if self._algorithm is not None:
            # Get algorithm's task selection preferences for tasks in this pool
            task_scores = self._algorithm.score_tasks(list(pool.keys()))
            if task_scores:
                # Convert scores to probabilities for sampling
                task_ids = list(task_scores.keys())
                scores = list(task_scores.values())
                total_score = sum(scores)
                if total_score > 0:
                    probabilities = [score / total_score for score in scores]
                    selected_id = self._rng.choices(task_ids, weights=probabilities)[0]
                    return pool[selected_id]

        # Fallback to random selection from pool
        return pool[self._rng.choice(list(pool.keys()))]

    def _create_task(self, pool: str) -> CurriculumTask:
        """Create a new task and add it to the specified pool.

        Args:
            pool: Which pool to add task to ("explore" or "exploit")
        """
        # Proactive integrity check before creation to prevent ID exhaustion
        pool_task_ids = set(self._explore_pool.keys()) | set(self._exploit_pool.keys())
        if pool_task_ids != self._task_ids:
            logger.warning(
                f"Task ID integrity issue detected before creation. "
                f"_task_ids size: {len(self._task_ids)}, pool IDs size: {len(pool_task_ids)}. "
                f"Auto-fixing to prevent ID exhaustion."
            )
            self._task_ids = pool_task_ids.copy()

        # Find unused task ID with collision detection and timeout
        max_attempts = 1000
        attempt = 0
        task_id = self._rng.randint(0, self._config.max_task_id)

        while task_id in self._task_ids:
            attempt += 1
            if attempt >= max_attempts:
                # Fallback: find any unused ID by scanning
                logger.warning(
                    f"Failed to find unused task_id after {max_attempts} random attempts. "
                    f"Current task_ids size: {len(self._task_ids)}, max_task_id: {self._config.max_task_id}. "
                    f"Scanning for unused ID..."
                )
                # Linear search for unused ID
                for candidate_id in range(self._config.max_task_id + 1):
                    if candidate_id not in self._task_ids:
                        task_id = candidate_id
                        break
                else:
                    # All IDs exhausted - this should never happen with reasonable config
                    raise RuntimeError(
                        f"Task ID space exhausted! Cannot create new task. "
                        f"task_ids size: {len(self._task_ids)}, max_task_id: {self._config.max_task_id}. "
                        f"Reduce num_active_tasks or increase max_task_id."
                    )
                break
            task_id = self._rng.randint(0, self._config.max_task_id)

        self._task_ids.add(task_id)
        env_cfg = self._task_generator.get_task(task_id)

        # Extract bucket values if available
        bucket_values = {}
        if hasattr(self._task_generator, "_last_bucket_values"):
            bucket_values = self._task_generator._last_bucket_values.copy()

        task = CurriculumTask(task_id, env_cfg, bucket_values)

        # Add to appropriate pool
        if pool == "explore":
            self._explore_pool[task_id] = task
        elif pool == "exploit":
            self._exploit_pool[task_id] = task
        else:
            raise ValueError(f"Invalid pool: {pool}. Must be 'explore' or 'exploit'")

        self._num_created += 1

        # Notify algorithm of new task
        if self._algorithm is not None and hasattr(self._algorithm, "on_task_created"):
            self._algorithm.on_task_created(task)

        return task

    def update_task_performance(self, task_id: int, score: float):
        """Update the curriculum algorithm with task performance."""
        if self._algorithm is not None:
            self._algorithm.update_task_performance(task_id, score)

        # Check for promotion
        self._check_promotion(task_id)

        # Invalidate stats cache since task performance affects curriculum stats
        self.invalidate_cache()

    def _check_promotion(self, task_id: int):
        """Check if a task from explore pool should be promoted to exploit pool."""
        # Only promote tasks from explore pool
        if task_id not in self._explore_pool:
            return

        task = self._explore_pool[task_id]

        # Check if task has reached promotion threshold
        if task._num_scheduled < self._config.promotion_threshold:
            return

        # Task is ready for promotion attempt
        self._num_promotions_attempted += 1

        # Get LPS scores for promotion and comparison
        worst_exploit_id_to_evict = None  # Track which task to evict after successful promotion

        if self._algorithm is None:
            # Without algorithm, use random promotion
            promote = self._rng.random() < 0.5
        else:
            # Find task with minimum LPS in exploit pool
            if not self._exploit_pool:
                # Exploit pool empty - auto-accept
                promote = True
            elif len(self._exploit_pool) < self._config.exploit_pool_capacity:
                # Exploit pool not full - auto-accept
                promote = True
            else:
                # Get scores for all tasks
                exploit_scores = self._algorithm.score_tasks(list(self._exploit_pool.keys()))
                promoted_task_score = self._algorithm.score_tasks([task_id]).get(task_id, 0.0)

                # Find worst task in exploit pool
                if exploit_scores:
                    worst_exploit_id = min(exploit_scores.keys(), key=lambda tid: exploit_scores[tid])
                    worst_exploit_score = exploit_scores[worst_exploit_id]

                    # Promote if promoted task has higher LPS than worst exploit task
                    if promoted_task_score > worst_exploit_score:
                        promote = True
                        # Mark worst task for eviction (will evict after successful promotion)
                        worst_exploit_id_to_evict = worst_exploit_id
                    else:
                        promote = False
                else:
                    # No scores available, use random
                    promote = self._rng.random() < 0.5

        # Execute promotion or rejection
        if promote:
            # Promotion accepted
            self._num_promotions_accepted += 1

            # First, try to create replacement explore task BEFORE modifying pools
            try:
                # Move task from explore to exploit pool
                self._explore_pool.pop(task_id)
                self._exploit_pool[task_id] = task

                # Create new task in explore pool to maintain capacity
                self._create_task(pool="explore")

                # Success - now evict worst task if we identified one earlier
                if worst_exploit_id_to_evict is not None:
                    self._evict_from_pool(worst_exploit_id_to_evict, "exploit")

                success_indicator = 1.0
            except Exception as e:
                logger.error(f"Failed to complete promotion: {e}")
                # Rollback: put the task back in explore if we moved it
                if task_id in self._exploit_pool:
                    self._exploit_pool.pop(task_id)
                    self._explore_pool[task_id] = task
                self._num_promotions_accepted -= 1
                success_indicator = 0.0
        else:
            # Promotion rejected
            # Discard task from explore pool
            self._evict_from_pool(task_id, "explore")
            # Create new task in explore pool
            try:
                self._create_task(pool="explore")
            except Exception as e:
                logger.error(f"Failed to create new explore task after rejection: {e}")
                # Put the task back to prevent pool from shrinking
                self._explore_pool[task_id] = task
                self._task_ids.add(task_id)
                self._num_evicted -= 1
            success_indicator = 0.0

        # Update P_accept with EMA
        self._P_accept = (1 - self._config.alpha) * self._P_accept + self._config.alpha * success_indicator

    def _evict_from_pool(self, task_id: int, pool: str):
        """Evict a task from a specific pool."""
        # Notify algorithm of eviction
        if self._algorithm is not None:
            self._algorithm.on_task_evicted(task_id)

        # Remove from appropriate pool
        task_was_in_pool = False
        if pool == "explore":
            if task_id in self._explore_pool:
                self._explore_pool.pop(task_id)
                task_was_in_pool = True
        elif pool == "exploit":
            if task_id in self._exploit_pool:
                self._exploit_pool.pop(task_id)
                task_was_in_pool = True

        if not task_was_in_pool:
            logger.warning(
                f"Attempted to evict task {task_id} from {pool} pool, but it wasn't there. "
                f"This may indicate a task management bug."
            )

        # Remove from global task IDs
        self._task_ids.discard(task_id)
        self._num_evicted += 1

    def get_base_stats(self) -> Dict[str, float]:
        """Get basic curriculum statistics."""
        all_tasks = list(self._explore_pool.values()) + list(self._exploit_pool.values())

        # Periodic integrity check (every ~10 stats calls to prevent ID exhaustion)
        if self._num_created % 10 == 0:
            pool_task_ids = set(self._explore_pool.keys()) | set(self._exploit_pool.keys())
            if pool_task_ids != self._task_ids:
                diff = (
                    self._task_ids - pool_task_ids
                    if len(self._task_ids) > len(pool_task_ids)
                    else pool_task_ids - self._task_ids
                )
                logger.error(
                    f"Task ID integrity check failed! "
                    f"_task_ids size: {len(self._task_ids)}, pool IDs size: {len(pool_task_ids)}. "
                    f"Difference: {diff}"
                )
                # Auto-fix: sync _task_ids with actual pools
                self._task_ids = pool_task_ids.copy()

        base_stats: Dict[str, float] = {
            "num_created": float(self._num_created),
            "num_evicted": float(self._num_evicted),
            "num_completed": float(sum(task._num_completions for task in all_tasks)),
            "num_scheduled": float(sum(task._num_scheduled for task in all_tasks)),
            "num_active_tasks": float(len(all_tasks)),
            "num_explore_tasks": float(len(self._explore_pool)),
            "num_exploit_tasks": float(len(self._exploit_pool)),
            "P_accept": float(self._P_accept),
            "num_promotions_attempted": float(self._num_promotions_attempted),
            "num_promotions_accepted": float(self._num_promotions_accepted),
            "promotion_rate": float(self._num_promotions_accepted / max(1, self._num_promotions_attempted)),
            "task_ids_size": float(len(self._task_ids)),  # Track for debugging
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
            "P_accept": self._P_accept,
            "num_promotions_attempted": self._num_promotions_attempted,
            "num_promotions_accepted": self._num_promotions_accepted,
            "explore_pool": {},
            "exploit_pool": {},
        }

        # Serialize explore pool tasks
        for task_id, task in self._explore_pool.items():
            state["explore_pool"][task_id] = {
                "num_completions": task._num_completions,
                "total_score": task._total_score,
                "mean_score": task._mean_score,
                "num_scheduled": task._num_scheduled,
                "slice_values": task._slice_values,
            }

        # Serialize exploit pool tasks
        for task_id, task in self._exploit_pool.items():
            state["exploit_pool"][task_id] = {
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

        # Restore two-pool state
        self._explore_pool.clear()
        self._exploit_pool.clear()
        self._task_ids.clear()

        self._P_accept = state.get("P_accept", 0.5)
        self._num_promotions_attempted = state.get("num_promotions_attempted", 0)
        self._num_promotions_accepted = state.get("num_promotions_accepted", 0)

        # Restore explore pool
        for task_id_str, task_data in state.get("explore_pool", {}).items():
            task_id = int(task_id_str)
            env_cfg = self._task_generator.get_task(task_id)
            task = CurriculumTask(task_id, env_cfg, task_data["slice_values"])
            task._num_completions = task_data["num_completions"]
            task._total_score = task_data["total_score"]
            task._mean_score = task_data["mean_score"]
            task._num_scheduled = task_data["num_scheduled"]

            self._explore_pool[task_id] = task
            self._task_ids.add(task_id)

        # Restore exploit pool
        for task_id_str, task_data in state.get("exploit_pool", {}).items():
            task_id = int(task_id_str)
            env_cfg = self._task_generator.get_task(task_id)
            task = CurriculumTask(task_id, env_cfg, task_data["slice_values"])
            task._num_completions = task_data["num_completions"]
            task._total_score = task_data["total_score"]
            task._mean_score = task_data["mean_score"]
            task._num_scheduled = task_data["num_scheduled"]

            self._exploit_pool[task_id] = task
            self._task_ids.add(task_id)

        # Restore algorithm state
        if self._algorithm is not None and "algorithm_state" in state:
            self._algorithm.load_state(state["algorithm_state"])


# Import concrete config classes at the end to avoid circular imports
# ruff: noqa: E402
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig

# Rebuild the model to resolve forward references
CurriculumConfig.model_rebuild()
