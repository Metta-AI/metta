"""Core curriculum orchestration and task pool management.

This module contains the main Curriculum class that orchestrates automatic curriculum
learning. The curriculum maintains a pool of active tasks, uses TaskGenerators to create
new tasks, and delegates to CurriculumAlgorithms for intelligent selection and eviction.

Key Components:
    - Curriculum: Main orchestrator managing the task pool lifecycle
    - CurriculumConfig: Configuration for curriculum behavior and algorithm selection
    - CurriculumTask: Wrapper holding task_id, env_cfg, and performance metrics
    - DiscreteRandomCurriculum: Simple reference algorithm for uniform random sampling

Separation of Concerns:
    - TaskGenerator creates environment configs (what tasks look like)
    - CurriculumAlgorithm decides which tasks to sample/evict (how to select)
    - Curriculum manages the pool and coordinates between them (orchestration)

Why This File:
    Central point of curriculum API and simple/reference implementations. Complex
    algorithms live in their own files (e.g., learning_progress_algorithm.py).

See Also:
    - learning_progress_algorithm.py: Learning progress implementation with bidirectional scoring
    - task_tracker.py: TaskTracker for managing task performance tracking
"""

from __future__ import annotations

import logging
import random
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import ConfigDict, Field

from metta.cogworks.curriculum.curriculum_base import (
    CurriculumAlgorithm,
    CurriculumAlgorithmConfig,
    CurriculumTask,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.stats import StatsLogger
from metta.cogworks.curriculum.task_generator import AnyTaskGeneratorConfig, SingleTaskGenerator
from mettagrid.base_config import Config
from mettagrid.config.mettagrid_config import MettaGridConfig

logger = logging.getLogger(__name__)


class DiscreteRandomCurriculumConfig(CurriculumAlgorithmConfig):
    """Hyperparameters for DiscreteRandomCurriculum."""

    type: Literal["discrete_random"] = "discrete_random"

    def algorithm_type(self) -> str:
        return "discrete_random"

    def create(self, num_tasks: int) -> CurriculumAlgorithm:
        return DiscreteRandomCurriculum(num_tasks, self)


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
    num_active_tasks: int = Field(default=1000, gt=0, description="Number of active tasks to maintain")

    # Curriculum behavior options
    seed: int = Field(default=0, description="Random seed for curriculum task generation")
    defer_init: bool = Field(default=False, description="Defer task pool initialization (used for checkpoint loading)")
    min_presentations_for_eviction: int = Field(
        default=5, gt=0, description="Minimum task presentations before eviction"
    )

    algorithm_config: Union["DiscreteRandomCurriculumConfig", "LearningProgressConfig"] = Field(
        default_factory=DiscreteRandomCurriculumConfig,
        description="Curriculum algorithm hyperparameters",
        discriminator="type",
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

        # Sync num_active_tasks between CurriculumConfig and algorithm_config
        # If algorithm_config has the default value (64), sync FROM CurriculumConfig
        # Otherwise, sync TO CurriculumConfig from algorithm_config (user explicitly set it)
        if self.algorithm_config.num_active_tasks == 64:  # Default from CurriculumAlgorithmConfig
            self.algorithm_config.num_active_tasks = self.num_active_tasks
        else:
            self.num_active_tasks = self.algorithm_config.num_active_tasks

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

    def __init__(self, config: CurriculumConfig):
        # Initialize StatsLogger (algorithm handles detailed stats)
        StatsLogger.__init__(self)

        self._config = config
        self._task_generator = config.task_generator.create()
        self._rng = random.Random(config.seed)
        self._tasks: dict[int, CurriculumTask] = {}
        self._task_ids: set[int] = set()
        self._num_created = 0
        self._num_evicted = 0

        # Create algorithm from config (always present via default_factory)
        num_tasks = config.algorithm_config.num_active_tasks
        self._algorithm: CurriculumAlgorithm = config.algorithm_config.create(num_tasks)
        # Pass curriculum reference to algorithm for stats updates
        self._algorithm.set_curriculum_reference(self)

        # Initialize task pool at capacity unless deferred (e.g., for checkpoint loading)
        if not config.defer_init:
            self._initialize_at_capacity()

    @property
    def _num_active_tasks(self) -> int:
        """Get the effective number of active tasks.

        Handles two scenarios:
        1. Algorithm_config in sync with config: use algorithm_config (allows CLI overrides)
        2. Algorithm_config out of sync: use config (manual override after creation)
        """
        # If they match, they're in sync - use algorithm_config (allows CLI overrides to work)
        if self._config.algorithm_config.num_active_tasks == self._config.num_active_tasks:
            return self._config.algorithm_config.num_active_tasks

        # If they differ, config was manually overridden - prefer config
        return self._config.num_active_tasks

    def get_task(self) -> CurriculumTask:
        """Sample a task from the population with replacement.

        Tasks are sampled with replacement - multiple environments can receive
        the same task_id simultaneously. The task remains in the pool after
        selection and is only removed via eviction.

        Returns:
            CurriculumTask: A task sampled from the pool based on learning progress scores
        """
        # Curriculum always manages the task pool - no delegation
        task = None
        if len(self._tasks) < self._num_active_tasks:
            task = self._create_task()

        # If we couldn't create a task, try eviction or choose existing
        if task is None:
            # At capacity - check if any task meets eviction criteria first
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

        # Notify algorithm that this task was sampled (for sampling statistics)
        self._algorithm.on_task_sampled(task._task_id)

        return task

    def _initialize_at_capacity(self) -> None:
        """Initialize the task pool to full capacity."""
        while len(self._tasks) < self._num_active_tasks:
            self._create_task()

    def _evict_specific_task(self, task_id: int) -> None:
        """Evict a specific task by ID."""
        if task_id not in self._tasks:
            return

        # Notify algorithm of eviction
        self._algorithm.on_task_evicted(task_id)

        self._task_ids.remove(task_id)
        self._tasks.pop(task_id)
        self._num_evicted += 1

    def _choose_task(self) -> CurriculumTask:
        """Choose a task from the population using algorithm guidance.

        Samples with replacement - the same task can be selected multiple times
        across different environments. Tasks are weighted by their learning
        progress scores (high LP = higher probability).
        """
        # Score all tasks
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
        """Create a new task with a unique ID from Python's unlimited integer space.

        Returns:
            The newly created CurriculumTask
        """
        # Use 53-bit integer space (2^53 - 1) to ensure compatibility with float64 storage
        # Float64 has 53 bits of mantissa precision, so integers up to 2^53 can be stored exactly
        # With ~1000 active tasks, collision probability is still negligible (~10^-13)
        task_id = self._rng.randint(0, 2**53 - 1)
        while task_id in self._task_ids:
            task_id = self._rng.randint(0, 2**53 - 1)
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
        self._algorithm.on_task_created(task)

        return task

    def update_task_performance(self, task_id: int, score: float):
        """Update the curriculum algorithm with task performance."""
        self._algorithm.update_task_performance(task_id, score)

        # Invalidate stats cache since task performance affects curriculum stats
        self.invalidate_cache()

    def get_task_lp_score(self, task_id: int) -> float:
        """Get the learning progress score for a specific task.

        Args:
            task_id: The task ID to get the score for

        Returns:
            The learning progress score, or 0.0 if not available
        """
        return self._algorithm.get_task_lp_score(task_id)

    def get_evictions_this_epoch(self) -> Dict[str, int]:
        """Get per-epoch evictions WITHOUT resetting the counter.

        Use this for reporting evictions in infos during episodes.

        Returns:
            Dictionary mapping label -> eviction count this epoch
        """
        return self._algorithm.get_evictions_this_epoch()

    def get_and_reset_evictions_this_epoch(self) -> Dict[str, int]:
        """Get per-epoch evictions and reset the counter.

        This should ONLY be called at epoch boundaries, not per-episode.

        Returns:
            Dictionary mapping label -> eviction count this epoch
        """
        return self._algorithm.get_and_reset_evictions_this_epoch()

    def reset_epoch_counters(self) -> None:
        """Reset per-epoch counters in the curriculum algorithm.

        This is called by the training infrastructure at epoch boundaries
        to ensure per-epoch metrics start fresh each epoch.
        """
        self._algorithm.reset_epoch_counters()

    def calculate_per_label_mean_lp_stats(self) -> Dict[str, float]:
        """Calculate per-label mean LP scores and reward EMAs.

        This is extracted from the main gini calculation and includes:
        - per_label_stats/mean_lp_score/{label}: Mean learning progress per label
        - per_label_stats/mean_reward_ema/{label}: Mean reward EMA per label (only completed tasks)

        Returns:
            Dictionary of per-label statistics
        """
        if not hasattr(self._algorithm, "task_tracker"):
            return {}

        stats = {}
        task_tracker = self._algorithm.task_tracker
        all_task_ids = task_tracker.get_all_tracked_tasks()

        if not all_task_ids:
            return stats

        # Group data by label
        label_lp_sums = {}
        label_lp_counts = {}
        label_reward_ema_sums = {}
        label_reward_ema_counts = {}

        for task_id in all_task_ids:
            task_stats = task_tracker.get_task_stats(task_id)
            if task_stats:
                label = task_tracker.get_task_label(task_id)
                if not label:
                    continue

                # Get raw LP score
                p_fast = float(task_stats.get("p_fast", 0.0))
                p_slow = float(task_stats.get("p_slow", 0.0))
                raw_lp = abs(p_fast - p_slow)

                label_lp_sums[label] = label_lp_sums.get(label, 0.0) + raw_lp
                label_lp_counts[label] = label_lp_counts.get(label, 0) + 1

                # Only include reward EMAs for tasks with completions
                completion_count = float(task_stats.get("completion_count", 0))
                if completion_count > 0:
                    reward_ema = float(task_stats.get("reward_ema", 0.0))
                    label_reward_ema_sums[label] = label_reward_ema_sums.get(label, 0.0) + reward_ema
                    label_reward_ema_counts[label] = label_reward_ema_counts.get(label, 0) + 1

        # Calculate means
        for label, lp_sum in label_lp_sums.items():
            count = label_lp_counts[label]
            mean_lp = lp_sum / count if count > 0 else 0.0
            stats[f"per_label_stats/mean_lp_score/{label}"] = mean_lp

        for label, reward_sum in label_reward_ema_sums.items():
            count = label_reward_ema_counts[label]
            mean_reward = reward_sum / count if count > 0 else 0.0
            stats[f"per_label_stats/mean_reward_ema/{label}"] = mean_reward

        return stats

    def calculate_raw_lp_debug_stats(self) -> Dict[str, float]:
        """Calculate raw LP debug statistics.

        This includes aggregate statistics about raw learning progress scores:
        - debug/raw_lp_mean: Mean of raw LP scores
        - debug/raw_lp_std: Standard deviation of raw LP scores
        - debug/raw_lp_min: Minimum raw LP score
        - debug/raw_lp_max: Maximum raw LP score
        - debug/raw_lp_nonzero_count: Count of tasks with non-zero LP
        - debug/raw_lp_total_count: Total number of tasks

        Returns:
            Dictionary of raw LP debug statistics
        """
        if not hasattr(self._algorithm, "task_tracker"):
            return {}

        stats = {}
        task_tracker = self._algorithm.task_tracker
        all_task_ids = task_tracker.get_all_tracked_tasks()

        if not all_task_ids:
            return stats

        raw_lp_scores = []
        for task_id in all_task_ids:
            task_stats = task_tracker.get_task_stats(task_id)
            if task_stats:
                p_fast = float(task_stats.get("p_fast", 0.0))
                p_slow = float(task_stats.get("p_slow", 0.0))
                raw_lp = abs(p_fast - p_slow)
                raw_lp_scores.append(raw_lp)

        if raw_lp_scores:
            import statistics

            mean_lp = statistics.mean(raw_lp_scores)
            stats["debug/raw_lp_mean"] = mean_lp
            stats["debug/raw_lp_std"] = statistics.stdev(raw_lp_scores) if len(raw_lp_scores) > 1 else 0.0
            stats["debug/raw_lp_min"] = min(raw_lp_scores)
            stats["debug/raw_lp_max"] = max(raw_lp_scores)
            stats["debug/raw_lp_nonzero_count"] = float(sum(1 for x in raw_lp_scores if x > 1e-10))
            stats["debug/raw_lp_total_count"] = float(len(raw_lp_scores))

        return stats

    def calculate_gini_coefficients(self) -> Dict[str, float]:
        """Calculate Gini coefficients at each stage of the LP calculation pipeline.

        This is an expensive operation that should be called once per epoch from
        the centralized stats reporter, not from each worker in vectorized environments.

        Calculates Gini coefficients for:
        - Raw LP scores (task-level and aggregated by label)
        - Z-scored LP scores
        - Sampling probabilities (task-level and by label)
        - Pool composition by label
        - Task ages
        - Eviction counts by label

        Returns:
            Dictionary of Gini coefficients with curriculum_gini/ prefix
        """
        return self._algorithm.calculate_gini_coefficients()

    def get_base_stats(self) -> Dict[str, float]:
        """Get basic curriculum statistics."""
        # Get global completion count from algorithm's task tracker if available
        # This tracks ALL completions across all tasks ever created (even evicted ones)
        per_label_completions: Dict[str, float] = {}
        if self._algorithm is not None and self._algorithm.task_tracker is not None:
            # CENTRALIZED MULTI-PROCESS COLLECTION:
            # get_all_tracked_tasks() scans shared memory to find ALL tasks from ALL worker processes
            # get_task_stats() reads completion counts directly from shared memory (synchronized)
            # This ensures we aggregate completions across ALL processes, not just one
            task_ids = self._algorithm.task_tracker.get_all_tracked_tasks()
            num_completed = 0.0
            for task_id in task_ids:
                stats = self._algorithm.task_tracker.get_task_stats(task_id)
                if stats:
                    completion_count = stats.get("completion_count", 0.0)
                    num_completed += completion_count
                    # Track per-label completions
                    label = self._algorithm.task_tracker.get_task_label(task_id)
                    if label:
                        per_label_completions[label] = per_label_completions.get(label, 0.0) + completion_count
        else:
            # Fallback: sum completions for currently active tasks only
            # NOTE: This undercounts if tasks have been evicted!
            num_completed = float(sum(task._num_completions for task in self._tasks.values()))
            for task in self._tasks.values():
                label = task.get_label()
                if label:
                    per_label_completions[label] = per_label_completions.get(label, 0.0) + float(task._num_completions)

        # Get num_active_tasks from task tracker if using shared memory, otherwise from local dict
        # This is critical when using shared memory - self._tasks may be sparse but TaskTracker
        # has the full set of active tasks across all workers
        if self._algorithm is not None and self._algorithm.task_tracker is not None:
            num_active_tasks = float(len(self._algorithm.task_tracker.get_all_tracked_tasks()))
        else:
            num_active_tasks = float(len(self._tasks))

        base_stats: Dict[str, float] = {
            "num_created": float(self._num_created),
            "num_evicted": float(self._num_evicted),
            "num_completed": num_completed,
            "num_scheduled": float(sum(task._num_scheduled for task in self._tasks.values())),
            "num_active_tasks": num_active_tasks,
        }

        # Add per-label completion counts
        for label, count in per_label_completions.items():
            base_stats[f"per_label_completions/{label}"] = count

        # Include algorithm stats
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

        # Save algorithm state
        state["algorithm_state"] = self._algorithm.get_state()

        num_tasks = len(state["tasks"])
        logger.info(f"Curriculum: Saving state with {num_tasks} tasks, algorithm_state=present")

        return state

    def log_shared_memory_state(self) -> None:
        """Log raw shared memory state for debugging.

        Directly reads the task tracker's shared memory arrays to show what's actually stored.
        This bypasses all caching and intermediate layers to provide ground truth.
        """
        if not hasattr(self._algorithm, "task_tracker"):
            logger.info("SHARED MEMORY DEBUG: No task tracker found")
            return

        task_tracker = self._algorithm.task_tracker
        logger.info("=" * 80)
        logger.info("SHARED MEMORY STATE (Direct Array Read)")
        logger.info("=" * 80)

        # Get all task IDs being tracked
        tracked_tasks = task_tracker.get_all_tracked_tasks()
        logger.info(f"Total tracked tasks: {len(tracked_tasks)}")

        if not tracked_tasks:
            logger.info("No tasks tracked yet")
            logger.info("=" * 80)
            return

        # Count tasks by completion status
        tasks_with_completions = 0
        total_completions = 0
        tasks_with_nonzero_lp = 0
        tasks_with_nonzero_ema = 0

        # Compute aggregate stats
        for task_id in tracked_tasks:
            task_stats = task_tracker.get_task_stats(task_id)
            if task_stats:
                comp_count = task_stats["completion_count"]
                if comp_count > 0:
                    tasks_with_completions += 1
                    total_completions += comp_count
                if task_stats["lp_score"] != 0.0:
                    tasks_with_nonzero_lp += 1
                if task_stats["p_fast"] != 0.0 or task_stats["p_slow"] != 0.0:
                    tasks_with_nonzero_ema += 1

        logger.info("\nAggregate stats:")
        logger.info(f"  Tasks with completions: {tasks_with_completions}/{len(tracked_tasks)}")
        logger.info(f"  Total completions across all tasks: {total_completions}")
        logger.info(f"  Tasks with non-zero LP scores: {tasks_with_nonzero_lp}/{len(tracked_tasks)}")
        logger.info(f"  Tasks with non-zero EMAs: {tasks_with_nonzero_ema}/{len(tracked_tasks)}")

        # Check global tracker stats
        tracker_global_stats = task_tracker.get_global_stats()
        logger.info("\nTracker global stats:")
        logger.info(f"  _total_completions: {tracker_global_stats.get('total_completions', 0)}")
        logger.info(f"  _mean_score: {tracker_global_stats.get('mean_score', 0):.4f}")

        logger.info("=" * 80)

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load curriculum state from checkpoint."""
        num_tasks_to_load = len(state.get("tasks", {}))
        has_algo_state = "algorithm_state" in state
        logger.info(
            f"Curriculum: Loading state with {num_tasks_to_load} tasks, "
            f"algorithm_state={'present' if has_algo_state else 'missing'}"
        )

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
        if "algorithm_state" in state:
            self._algorithm.load_state(state["algorithm_state"])

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

        logger.info(f"Curriculum: Successfully loaded {len(self._tasks)} tasks")

        # NOTE: We don't call on_task_created() here because:
        # 1. Algorithm state (including task_tracker) is already restored above via load_state()
        # 2. Calling it would re-initialize tracking with default values


# Rebuild the model to resolve forward references
CurriculumConfig.model_rebuild()
