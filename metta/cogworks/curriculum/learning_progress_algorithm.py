"""Learning progress curriculum algorithm.

This module implements the Learning Progress (LP) algorithm - a curriculum learning approach
that prioritizes tasks where the agent is learning fastest. It tracks fast/slow EMAs of task
performance to identify learning opportunities, then samples tasks proportionally to their
learning progress scores.

Core Algorithm:
    1. Track task performance using exponential moving averages (fast and slow)
    2. Compute learning progress as the rate of performance change (|fast - slow|)
    3. Apply exploration bonus to under-sampled tasks
    4. Transform scores with z-score normalization and sigmoid for sampling probabilities
    5. Sample tasks proportionally to their scores (high LP → more likely to be selected)

Key Components:
    - LearningProgressConfig: Comprehensive configuration with sensible defaults
    - LearningProgressAlgorithm: Main algorithm coordinating scorer, tracker, and stats
    - TaskTracker: Manages task performance tracking
    - LPScorer: Strategy pattern for bidirectional/basic LP scoring

Design Philosophy:
    - All state (EMAs, counts) lives in shared memory for true multi-process training
    - Strategy pattern for scoring allows swapping between bidirectional/basic/custom
    - Stateless algorithms make checkpointing and debugging straightforward

Configuration Helpers:
    - LearningProgressConfig.default(): Balanced config
    - LearningProgressConfig.stable(): Stable config for noisy environments
    - LearningProgressConfig.fast_learning(): Fast adaptation for quick learners

See Also:
    - task_tracker.py: TaskTracker for performance tracking
    - lp_scorers.py: Scoring strategies (bidirectional/basic LP)
    - curriculum.py: Main Curriculum class using this algorithm
"""

import logging
import math
import random
import statistics
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import model_validator

from .curriculum_base import CurriculumAlgorithm, CurriculumAlgorithmConfig, CurriculumTask
from .lp_scorers import BasicLPScorer, BidirectionalLPScorer, LPScorer
from .stats import StatsLogger
from .task_tracker import TaskTracker

logger = logging.getLogger(__name__)


class LearningProgressConfig(CurriculumAlgorithmConfig):
    """Configuration for learning progress with bidirectional scoring as default."""

    type: Literal["learning_progress"] = "learning_progress"

    # Bidirectional learning progress settings (now default)
    use_bidirectional: bool = True
    use_baseline_normalization: bool = (
        True  # Normalize by baseline to get "mastery" score p_i = (TSR_i - B_i) / (1.0 - B_i)
    )
    # EMA Timescale: Controls convergence speed of fast EMA
    # - 0.1 (default): Converges in ~10 samples, responsive to recent changes
    # - 0.01-0.05: Slower convergence, more stable for noisy environments
    # - 0.001: Very slow (1000+ samples), delays LP signal but maximum stability
    # Lower values delay learning progress signal development - Gini may stay near 0
    ema_timescale: float = 0.1
    slow_timescale_factor: float = 0.2  # Multiplier for slow EMA timescale (slow = ema_timescale * this)
    exploration_bonus: float = 0.1
    progress_smoothing: float = 0.0  # For bidirectional reweighting (set to 0 to avoid artificial floor)
    performance_bonus_weight: float = 0.0  # Weight for performance bonus in LP calculation
    lp_score_temperature: float = 0.0  # Temperature for rescaling LP scores before sigmoid
    # Special values for lp_score_temperature:
    # - > 0: Divide LP by temperature (low temp amplifies differences)
    # - = 0: Apply z-score normalization (standardize to mean=0, std=1) before sigmoid (DEFAULT)
    #        This centers LP scores and makes sigmoid more sensitive to relative differences
    z_score_amplification: float = 10.0  # Amplification factor after z-score normalization
    # Only applies when lp_score_temperature = 0 (z-score mode). Higher values increase selectivity
    # by spreading out the z-scored distribution before sigmoid. Default 10.0 provides strong selectivity
    # while maintaining z-score's scale-invariance. Set to 1.0 for no amplification (uniform sampling).
    early_progress_amplification: float = 0.5  # Reweight performance signals before LP calculation
    # Note: 0.5 is effectively OFF (R(p) ≈ p). Low values (e.g., 0.05) amplify signal from
    # unsolved tasks (p~0) and dampen signal from partially-solved tasks (p~0.5).
    # High values would reweight toward higher performance tasks.

    # Task distribution and sampling
    num_active_tasks: int = 1000
    rand_task_rate: float = 0.01  # Reduced from 0.25 in refactor for better curriculum learning
    sample_threshold: int = 10
    memory: int = 25
    eviction_threshold_percentile: float = 0.4  # Bottom percentile for task eviction

    # Memory management for label tracking
    max_inactive_labels_retained: int = 100  # Max inactive labels to keep for historical stats (prevents memory leak)

    # Basic EMA mode parameters (when use_bidirectional=False)
    basic_ema_initial_alpha: float = 0.3  # Initial learning rate for basic EMA
    basic_ema_alpha_decay: float = 0.2  # Decay factor for basic EMA alpha
    min_samples_for_lp: int = 10  # Minimum samples before using LP score (use exploration bonus until then)

    # Task tracker EMA configuration
    task_tracker_ema_alpha: float = 0.02  # Learning rate for task tracker EMAs (reward, success rate)

    # Task creation defaults
    task_default_success_threshold: float = 0.5  # Default success threshold for new tasks
    task_default_generator_type: float = 0.0  # Default generator type identifier for tasks

    # Memory backend configuration
    task_struct_size: int = 18  # Size of task data structure in shared memory (17 metrics + label_hash)
    use_shared_memory: bool = True  # Enabled by default for production use
    session_id: Optional[str] = None  # Session ID for shared memory, None = auto-generate shared ID

    # Logging configuration
    show_curriculum_troubleshooting_logging: bool = False  # Show high-cardinality per-task metrics for debugging

    @model_validator(mode="after")
    def _validate_and_initialize(self) -> "LearningProgressConfig":
        """Validate configuration and initialize derived parameters.

        This ensures session ID is generated when using shared memory.
        """
        # Generate session ID for shared memory if not provided
        if self.use_shared_memory and self.session_id is None:
            # Generate a unique session ID that will be shared across processes
            # This happens once at config creation time, before pickling
            self.session_id = f"lp_{uuid.uuid4().hex[:8]}"

        return self

    def algorithm_type(self) -> str:
        return "learning_progress"

    def create(self, num_tasks: int, stats_logger: "StatsLogger") -> "LearningProgressAlgorithm":
        return LearningProgressAlgorithm(num_tasks, stats_logger, self)

    # Configuration Presets for Common Use Cases
    # These provide sensible defaults for different training scenarios

    @classmethod
    def default(cls, num_active_tasks: int = 256, **overrides) -> "LearningProgressConfig":
        """Standard configuration with balanced learning speed.

        Best for: Most RL environments with moderate complexity
        - Bidirectional LP for intelligent task selection
        - Fast EMA convergence (~10 samples)
        - Strong z-score amplification for selectivity

        Args:
            num_active_tasks: Number of tasks to keep in active pool
            **overrides: Override any parameter
        """
        defaults = {
            "use_bidirectional": True,
            "ema_timescale": 0.1,
            "num_active_tasks": num_active_tasks,
            "slow_timescale_factor": 0.2,
            "rand_task_rate": 0.01,
            "exploration_bonus": 0.1,
            "min_samples_for_lp": 10,
            "lp_score_temperature": 0.0,
            "z_score_amplification": 10.0,
            "show_curriculum_troubleshooting_logging": False,
            "early_progress_amplification": 0.5,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def stable(cls, num_active_tasks: int = 256, **overrides) -> "LearningProgressConfig":
        """Stable configuration for noisy/stochastic environments.

        Best for: Environments with high variance or randomness
        - Slower EMA convergence for stability (~100 samples)
        - Higher exploration bonus
        - More gradual learning progress signal development

        Args:
            num_active_tasks: Number of tasks to keep in active pool
            **overrides: Override any parameter
        """
        defaults = {
            "use_bidirectional": True,
            "ema_timescale": 0.01,  # 10x slower
            "num_active_tasks": num_active_tasks,
            "slow_timescale_factor": 0.2,
            "rand_task_rate": 0.02,  # More exploration
            "exploration_bonus": 0.15,  # Higher exploration
            "min_samples_for_lp": 20,  # More samples before LP
            "lp_score_temperature": 0.0,
            "z_score_amplification": 5.0,  # Less aggressive
            "show_curriculum_troubleshooting_logging": False,
            "early_progress_amplification": 0.5,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def fast_learning(cls, num_active_tasks: int = 256, **overrides) -> "LearningProgressConfig":
        """Fast learning configuration for quickly-adapting agents.

        Best for: Simple environments where agent learns rapidly
        - Very fast EMA convergence (~5 samples)
        - Low exploration bonus (focus on LP)
        - Strong selectivity for high-LP tasks

        Args:
            num_active_tasks: Number of tasks to keep in active pool
            **overrides: Override any parameter
        """
        defaults = {
            "use_bidirectional": True,
            "ema_timescale": 0.2,  # 2x faster
            "num_active_tasks": num_active_tasks,
            "slow_timescale_factor": 0.2,
            "rand_task_rate": 0.005,  # Less exploration
            "exploration_bonus": 0.05,  # Lower exploration
            "min_samples_for_lp": 5,  # Quick LP signal
            "lp_score_temperature": 0.0,
            "z_score_amplification": 15.0,  # More aggressive
            "show_curriculum_troubleshooting_logging": False,
            "early_progress_amplification": 0.5,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def arena_legacy(cls, num_active_tasks: int = 256, **overrides) -> "LearningProgressConfig":
        """Legacy arena configuration (from before refactor).

        Best for: Reproducing old arena training runs
        - Very slow EMA for maximum stability (1000+ samples)
        - High z-score amplification

        Args:
            num_active_tasks: Number of tasks to keep in active pool
            **overrides: Override any parameter
        """
        defaults = {
            "use_bidirectional": True,
            "ema_timescale": 0.001,  # Very slow (legacy setting)
            "num_active_tasks": num_active_tasks,
            "slow_timescale_factor": 0.2,
            "rand_task_rate": 0.01,
            "exploration_bonus": 0.1,
            "min_samples_for_lp": 10,
            "lp_score_temperature": 0.0,
            "z_score_amplification": 10.0,
            "show_curriculum_troubleshooting_logging": True,
            "early_progress_amplification": 0.5,
        }
        defaults.update(overrides)
        return cls(**defaults)


class LearningProgressAlgorithm(CurriculumAlgorithm):
    """
    Learning Progress Algorithm with integrated bidirectional scoring.

    Uses bidirectional learning progress by default, combining fast and slow
    exponential moving averages to detect learning opportunities and guide
    intelligent task selection.
    """

    def __init__(self, num_tasks: int, stats_logger: "StatsLogger", hypers: LearningProgressConfig):
        super().__init__(num_tasks, stats_logger, hypers)

        self.num_tasks = num_tasks
        self.hypers: LearningProgressConfig = hypers

        # Initialize task tracker
        self.task_tracker = TaskTracker(
            max_memory_tasks=hypers.num_active_tasks,
            ema_alpha=hypers.task_tracker_ema_alpha,
            session_id=hypers.session_id if hypers.use_shared_memory else None,
            use_shared_memory=hypers.use_shared_memory,
            task_struct_size=hypers.task_struct_size,
            default_success_threshold=hypers.task_default_success_threshold,
            default_generator_type=hypers.task_default_generator_type,
        )

        # Initialize scorer strategy (pass tracker for shared memory EMA access)
        self.scorer: LPScorer = (
            BidirectionalLPScorer(hypers, self.task_tracker)
            if hypers.use_bidirectional
            else BasicLPScorer(hypers, self.task_tracker)
        )

        # Track label sampling and eviction (labels themselves are in TaskTracker shared memory)
        self._label_sampling_counts: Dict[str, int] = {}  # label -> cumulative sampling count (episodes started)
        self._label_eviction_counts: Dict[str, int] = {}  # label -> eviction count (cumulative)

        # Per-epoch tracking (for gini calculation and epoch-level metrics)
        self._label_evictions: Dict[str, int] = {}  # label -> evictions
        self._label_sampling_counts: Dict[str, int] = {}  # label -> samples

        # Track which labels are currently active (have tasks in pool)
        self._active_labels: set[str] = set()

        # Track recently inactive labels to manage memory
        self._inactive_labels_fifo: list[str] = []  # FIFO queue of inactive labels for cleanup

    def stats(self, prefix: str = "") -> Dict[str, float]:
        """Get all statistics with optional prefix. Always includes learning progress stats."""
        # Use the StatsLogger implementation
        return super().stats(prefix)

    def score_tasks(self, task_ids: List[int]) -> Dict[int, float]:
        """Score tasks using the configured method (bidirectional by default)."""
        return {task_id: self.scorer.score_task(task_id, self.task_tracker) for task_id in task_ids}

    def recommend_eviction(self, all_task_ids: List[int], min_presentations: int) -> Optional[int]:
        """Recommend which task to evict based on learning progress.

        Args:
            all_task_ids: All active task IDs in the pool
            min_presentations: Minimum presentations required before a task can be evicted

        Returns:
            Task ID to evict, or None if no task meets eviction criteria
        """
        if not all_task_ids:
            return None

        # Filter to evictable tasks
        evictable_tasks = [tid for tid in all_task_ids if self.should_evict_task(tid, min_presentations)]

        if not evictable_tasks:
            return None

        scores = self.score_tasks(evictable_tasks)

        # Find task with minimum learning progress
        min_task_id = min(evictable_tasks, key=lambda tid: scores.get(tid, 0.0))
        return min_task_id

    def should_evict_task(self, task_id: int, min_presentations: int = 5) -> bool:
        """Check if a task should be evicted based on criteria."""
        # First check if task has enough presentations
        task_stats = self.task_tracker.get_task_stats(task_id)
        if task_stats is None:
            return False

        if task_stats["completion_count"] < min_presentations:
            return False

        # Check if this task has low learning progress compared to others
        all_task_ids = self.task_tracker.get_all_tracked_tasks()
        if len(all_task_ids) <= 1:
            return False

        scores = self.score_tasks(all_task_ids)
        task_score = scores.get(task_id, 0.0)

        # Evict if this task is in the bottom N% of learning progress scores
        # This ensures eviction happens more readily with small task pools
        sorted_scores = sorted(scores.values())
        threshold_index = max(0, int(len(sorted_scores) * self.hypers.eviction_threshold_percentile))
        threshold_score = sorted_scores[threshold_index] if sorted_scores else 0.0

        return task_score <= threshold_score

    def on_task_evicted(self, task_id: int) -> None:
        """Clean up when a task is evicted."""
        # Get label BEFORE removing task (otherwise data is gone)
        evicted_label = self.task_tracker.get_task_label(task_id)

        # Remove from task tracker (handles its own locking)
        self.task_tracker.remove_task(task_id)

        # Learning progress specific cleanup
        self._remove_task_from_scoring(task_id)

        # Track eviction by label
        # Track cumulative eviction count for this label
        self._label_eviction_counts[evicted_label] = self._label_eviction_counts.get(evicted_label, 0) + 1

        # Track per-epoch eviction count (for gini calculation)
        self._label_evictions[evicted_label] = self._label_evictions.get(evicted_label, 0) + 1

        # Check if this label still has any active tasks
        # get_all_tracked_tasks() only returns ACTIVE tasks, so this is safe
        all_active_labels = set()
        for tid in self.task_tracker.get_all_tracked_tasks():
            label = self.task_tracker.get_task_label(tid)
            all_active_labels.add(label)

        if evicted_label not in all_active_labels:
            # No more tasks with this label - remove from active set and track as inactive
            self._active_labels.discard(evicted_label)
            self._inactive_labels_fifo.append(evicted_label)

            # Clean up old inactive labels to prevent memory leak
            self._cleanup_old_inactive_labels()

        # Invalidate stats cache when task state changes
        self.invalidate_cache()

    def _cleanup_old_inactive_labels(self) -> None:
        """Clean up old inactive labels to prevent unbounded memory growth.

        Keeps only the most recent N inactive labels as specified by
        max_inactive_labels_retained config parameter.
        """
        max_retained = self.hypers.max_inactive_labels_retained

        # Remove old labels if we exceed the limit
        while len(self._inactive_labels_fifo) > max_retained:
            old_label = self._inactive_labels_fifo.pop(0)

            # Only clean up if this label is still inactive (not reactivated)
            if old_label not in self._active_labels:
                # Clean up cumulative stats for this label
                # (completion counts are now in TaskTracker shared memory, no cleanup needed)
                self._label_sampling_counts.pop(old_label, None)
                self._label_eviction_counts.pop(old_label, None)

                # Note: We don't clean up per-epoch counters here as they're reset each epoch anyway

    def _remove_task_from_scoring(self, task_id: int) -> None:
        """Remove task from scoring system."""
        self.scorer.remove_task(task_id)

    def on_task_sampled(self, task_id: int) -> None:
        """Track that a task was sampled (selected for an episode).

        Args:
            task_id: The ID of the task that was sampled
        """
        # Track sampling counts per label (both cumulative and per-epoch)
        label = self.task_tracker.get_task_label(task_id)
        self._label_sampling_counts[label] = self._label_sampling_counts.get(label, 0) + 1
        self._label_sampling_counts[label] = self._label_sampling_counts.get(label, 0) + 1

    def get_evictions(self) -> Dict[str, int]:
        """Get evictions WITHOUT resetting the counter.

        Use this for reporting evictions in infos during episodes.

        Returns:
            Dictionary mapping label -> eviction count
        """
        return self._label_evictions.copy()

    def get_and_reset_evictions(self) -> Dict[str, int]:
        """Get evictions and reset the counter.

        This should ONLY be called at epoch boundaries, not per-episode.
        For per-episode reporting, use get_evictions() instead.

        Returns:
            Dictionary mapping label -> eviction count
        """
        evictions = self._label_evictions.copy()
        self._label_evictions.clear()
        return evictions

    def get_and_reset_sampling_counts(self) -> Dict[str, int]:
        """Get sampling counts and reset the counter.

        Returns:
            Dictionary mapping label -> sampling count
        """
        sampling_counts = self._label_sampling_counts.copy()
        self._label_sampling_counts.clear()
        return sampling_counts

    def on_epoch_end(self) -> None:
        """Handle epoch end event.

        This is called by the training infrastructure at epoch boundaries
        to ensure metrics start fresh.
        """
        self._label_sampling_counts.clear()
        self._label_evictions.clear()

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance atomically.

        Stage 3 Atomic Update: All EMA updates happen in ONE lock acquisition:
        1. Basic EMAs (completion_count, reward_ema, success_rate_ema, ema_squared)
        2. Bidirectional EMAs (p_fast, p_slow, p_true, random_baseline)

        LP score calculation is deferred until sampling time (lazy evaluation via _stale_dist flag).
        This reduces from 4+ lock acquisitions to 1.

        """
        # Atomic update: All EMAs in one lock
        self.task_tracker.update_task_performance_with_bidirectional_emas(
            task_id=task_id,
            score=score,
            scorer=self.scorer if hasattr(self.scorer, "config") else None,
        )

        # Mark distribution as stale - LP scores will be recalculated on next sampling
        self.scorer.invalidate_cache()

        # Invalidate stats cache when task performance changes
        self.invalidate_cache()

    def _choose_task_from_list(self, task_ids: List[int]) -> int:
        """Choose a task from the provided list based on scores."""
        if not task_ids:
            raise ValueError("Cannot choose from empty task list")

        scores = self.score_tasks(task_ids)
        if not scores:
            return random.choice(task_ids)

        # Convert scores to probabilities for sampling
        total_score = sum(scores.values())
        if total_score <= 0:
            return random.choice(task_ids)

        # Create weighted probability distribution
        weights = [scores.get(task_id, 0.0) for task_id in task_ids]
        return random.choices(task_ids, weights=weights)[0]

    def get_task_score(self, task_id: int) -> float:
        """Get the score for a specific task (sampling probability)."""
        return self.scorer.score_task(task_id, self.task_tracker)

    def on_task_created(self, task: CurriculumTask) -> None:
        """Handle task creation by tracking it.

        Args:
            task: The curriculum task being created
        """
        # Track the task
        self.task_tracker.track_task_creation(task._task_id)

        # Check if task was actually tracked (might fail if tracker is full)
        if task._task_id not in self.task_tracker._task_id_to_index:
            # Task wasn't tracked (tracker is full), don't add label
            return

        # Initialize LP score to exploration bonus for new tasks
        self.task_tracker.update_lp_score(task._task_id, self.hypers.exploration_bonus)

        # Handle label tracking
        # Note: task.get_label() always returns a string (defaults to "unknown")
        label = task.get_label()

        # Store label in TaskTracker's shared memory
        self.task_tracker.set_task_label(task._task_id, label)

        # If label was inactive, remove it from the inactive queue (reactivating it)
        if label in self._inactive_labels_fifo:
            self._inactive_labels_fifo.remove(label)

        self._active_labels.add(label)

        # Invalidate stats cache when task state changes
        self.invalidate_cache()

    def get_pool_composition_stats(self) -> Dict[str, Dict[str, int]]:
        """Get pool composition and sampling statistics by label.

        Returns:
            Dictionary with 'pool_composition' and 'sampling_counts' keys,
            each containing label->count mappings.
        """
        # Count labels currently in pool from TaskTracker shared memory
        pool_composition = {}
        for task_id in self.task_tracker.get_all_tracked_tasks():
            label = self.task_tracker.get_task_label(task_id)
            pool_composition[label] = pool_composition.get(label, 0) + 1

        # Return sampling counts (reset each epoch)
        return {
            "pool_composition": pool_composition,
            "sampling_counts": self._label_sampling_counts.copy(),
        }

    def get_base_stats(self) -> Dict[str, float]:
        """Get basic statistics that all algorithms must provide.

        Note: Called per-worker in vectorized environments, so keep lightweight.
        Expensive calculations like Gini are in calculate_gini_coefficients().
        """
        # Start with number of tasks
        stats = {
            "num_tasks": self.num_tasks,
        }

        # Add task tracker global stats with prefix
        tracker_stats = self.task_tracker.get_global_stats()
        for key, value in tracker_stats.items():
            stats[f"tracker/{key}"] = value

        # Add pool composition and sampling statistics
        composition_data = self.get_pool_composition_stats()

        for label, count in composition_data["pool_composition"].items():
            stats[f"pool_composition/{label}"] = float(count)

        for label, count in composition_data["sampling_counts"].items():
            stats[f"sampling_counts/{label}"] = float(count)

        for label, count in self._label_eviction_counts.items():
            stats[f"eviction_counts/{label}"] = float(count)

        return stats

    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for a distribution.

        Measures inequality in sampling/distribution:
        - 0 = perfect equality (all values equal)
        - 1 = perfect inequality (one value has everything)

        Args:
            values: List of counts/frequencies

        Returns:
            Gini coefficient between 0 and 1
        """
        if not values or len(values) == 0:
            return 0.0

        # Handle case with all zeros
        total = sum(values)
        if total == 0:
            return 0.0

        # Check for NaN or inf
        if any(math.isnan(v) or math.isinf(v) for v in values):
            logger.warning(f"Gini calculation received NaN or Inf values: {values[:10]}...")
            return 0.0

        # Sort values in ascending order
        sorted_values = sorted(values)
        n = len(sorted_values)

        # Calculate Gini coefficient using the formula:
        # G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
        cumsum = 0.0
        for i, value in enumerate(sorted_values, start=1):
            cumsum += i * value

        gini = (2.0 * cumsum) / (n * total) - (n + 1.0) / n

        # Sanity check result
        if math.isnan(gini) or math.isinf(gini):
            logger.error(f"Gini calculation produced NaN/Inf! cumsum={cumsum}, n={n}, total={total}")
            return 0.0

        return gini

    def calculate_gini_coefficients(self) -> Dict[str, float]:
        """Calculate Gini coefficients at each stage of the LP calculation pipeline.

        This is an expensive operation that should be called once per epoch from
        the centralized stats reporter, not from each worker in vectorized environments.

        This helps diagnose where selectivity is lost in the chain:
        1. Raw LP scores (task-level)
        2. Raw LP scores aggregated by label
        3. Z-scored LP scores (task-level)
        4. Final sampling probabilities (task-level)
        5. Sampling counts aggregated by label
        6. Eviction counts aggregated by label
        7. Pool composition aggregated by label

        Returns:
            Dictionary of Gini coefficients at each pipeline stage
        """
        gini_stats = {}

        # Get all tracked tasks
        all_task_ids = self.task_tracker.get_all_tracked_tasks()

        if not all_task_ids:
            return gini_stats

        # Collect task-level data
        completion_counts = []
        raw_lp_scores = []  # Actual raw LP: |p_fast - p_slow|
        z_scored_lp_scores = []
        sampling_probs = []
        task_labels_list = []
        task_ages = []

        current_time = time.time()

        for task_id in all_task_ids:
            task_stats = self.task_tracker.get_task_stats(task_id)
            completion_count = float(task_stats["completion_count"])
            completion_counts.append(completion_count)

            # Get final sampling probability (after all transformations)
            sampling_prob = self.scorer.score_task(task_id, self.task_tracker)
            sampling_probs.append(float(sampling_prob))
            z_scored_lp_scores.append(float(sampling_prob))  # Currently same as sampling_prob

            # Get actual raw learning progress: |p_fast - p_slow|
            # This is the true LP signal before smoothing/normalization
            p_fast = float(task_stats.get("p_fast", 0.0))
            p_slow = float(task_stats.get("p_slow", 0.0))
            raw_lp = abs(p_fast - p_slow)
            raw_lp_scores.append(raw_lp)

            # Calculate task age
            creation_time = float(task_stats.get("creation_time", current_time))
            task_age = current_time - creation_time
            task_ages.append(task_age)

            label = self.task_tracker.get_task_label(task_id)
            task_labels_list.append(label)

            if self.hypers.show_curriculum_troubleshooting_logging:
                gini_stats[f"task_metrics/{task_id}/completion_count"] = completion_count
                gini_stats[f"task_metrics/{task_id}/raw_lp"] = raw_lp
                gini_stats[f"task_metrics/{task_id}/sampling_prob"] = sampling_prob

        if completion_counts:
            gini_stats["curriculum_gini/pool_occupancy"] = self._calculate_gini_coefficient(completion_counts)

        if task_ages:
            gini_stats["curriculum_gini/task_age"] = self._calculate_gini_coefficient(task_ages)
            mean_age = statistics.mean(task_ages)
            gini_stats["debug/task_age_mean_seconds"] = mean_age
            gini_stats["debug/task_age_max_seconds"] = max(task_ages)

        if raw_lp_scores:
            gini_stats["curriculum_gini/raw_lp_scores"] = self._calculate_gini_coefficient(raw_lp_scores)
            # Note: Raw LP debug stats (mean, std, min, max, etc.) are now calculated
            # separately via curriculum.calculate_raw_lp_debug_stats()

        if raw_lp_scores and task_labels_list:
            # Calculate label-aggregated raw LP for gini calculation
            label_lp_sums = {}
            for i, label in enumerate(task_labels_list):
                lp = raw_lp_scores[i]
                label_lp_sums[label] = label_lp_sums.get(label, 0.0) + lp

            if label_lp_sums:
                label_lp_values = list(label_lp_sums.values())
                gini_stats["curriculum_gini/raw_lp_by_label"] = self._calculate_gini_coefficient(label_lp_values)
                # Note: Per-label mean LP scores and reward EMAs are now calculated
                # separately via curriculum.calculate_per_label_mean_lp_stats()

        if z_scored_lp_scores:
            gini_stats["curriculum_gini/zscored_lp_scores"] = self._calculate_gini_coefficient(
                [abs(z) for z in z_scored_lp_scores]
            )

        if sampling_probs:
            gini_stats["curriculum_gini/sampling_probs"] = self._calculate_gini_coefficient(sampling_probs)

        if sampling_probs and task_labels_list:
            label_prob_sums = {}
            for label, prob in zip(task_labels_list, sampling_probs, strict=True):
                label_prob_sums[label] = label_prob_sums.get(label, 0.0) + prob

            if label_prob_sums:
                label_prob_values = list(label_prob_sums.values())
                gini_stats["curriculum_gini/sampling_probs_by_label"] = self._calculate_gini_coefficient(
                    label_prob_values
                )

        if self._label_sampling_counts:
            label_sampling_values = list(self._label_sampling_counts.values())
            gini_stats["curriculum_gini/sampling_by_label"] = self._calculate_gini_coefficient(label_sampling_values)

        if self._label_eviction_counts:
            label_eviction_values = list(self._label_eviction_counts.values())
            if label_eviction_values and sum(label_eviction_values) > 0:
                gini_stats["curriculum_gini/evictions_by_label"] = self._calculate_gini_coefficient(
                    label_eviction_values
                )

        composition_data = self.get_pool_composition_stats()
        if composition_data["pool_composition"]:
            pool_comp_values = list(composition_data["pool_composition"].values())
            gini_stats["curriculum_gini/pool_composition_by_label"] = self._calculate_gini_coefficient(pool_comp_values)

        # Selectivity loss: how much inequality is reduced in the transformation pipeline
        if "curriculum_gini/raw_lp_scores" in gini_stats and "curriculum_gini/sampling_probs" in gini_stats:
            selectivity_loss = (
                gini_stats["curriculum_gini/raw_lp_scores"] - gini_stats["curriculum_gini/sampling_probs"]
            )
            gini_stats["curriculum_gini/selectivity_loss_lp_to_prob"] = selectivity_loss

        if "curriculum_gini/raw_lp_by_label" in gini_stats and "curriculum_gini/sampling_probs_by_label" in gini_stats:
            label_prob_selectivity_loss = (
                gini_stats["curriculum_gini/raw_lp_by_label"] - gini_stats["curriculum_gini/sampling_probs_by_label"]
            )
            gini_stats["curriculum_gini/selectivity_loss_lp_label_to_prob_label"] = label_prob_selectivity_loss

        if "curriculum_gini/raw_lp_by_label" in gini_stats and "curriculum_gini/sampling_by_label" in gini_stats:
            label_selectivity_loss = (
                gini_stats["curriculum_gini/raw_lp_by_label"] - gini_stats["curriculum_gini/sampling_by_label"]
            )
            gini_stats["curriculum_gini/selectivity_loss_lp_label_to_sampling_label"] = label_selectivity_loss

        return gini_stats

    def get_detailed_stats(self) -> Dict[str, float]:
        """Get detailed stats including learning progress and slice distribution analysis."""
        stats = {}

        # Learning progress stats from scorer with lp/ prefix
        lp_stats = self.scorer.get_stats()
        for key, value in lp_stats.items():
            stats[f"lp/{key}"] = value

        return stats

    def get_state(self) -> Dict[str, Any]:
        """Get learning progress algorithm state for checkpointing."""
        state = {
            "type": self.hypers.algorithm_type(),
            "hypers": self.hypers.model_dump(),
            "task_tracker": self.task_tracker.get_state(),
            "scorer": self.scorer.get_state(),
            "label_tracking": {
                # Labels are now stored in TaskTracker shared memory
                # Only save sampling/eviction counts and active label metadata
                "label_sampling_counts": self._label_sampling_counts,
                "label_eviction_counts": self._label_eviction_counts,
                "active_labels": list(self._active_labels),
                "inactive_labels_fifo": self._inactive_labels_fifo,
            },
        }

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load learning progress algorithm state from checkpoint."""
        # Restore task tracker
        self.task_tracker.load_state(state["task_tracker"])

        # Log what was restored
        num_tasks = len(self.task_tracker.get_all_tracked_tasks())
        total_completions = self.task_tracker._total_completions
        logger.info(
            f"LP Algorithm: Loaded {num_tasks} tasks from checkpoint with {total_completions} total completions"
        )

        # Restore scorer state
        if "scorer" in state:
            self.scorer.load_state(state["scorer"])

        # Restore label tracking state (if available, for backward compatibility)
        # Labels themselves are now in TaskTracker shared memory
        if "label_tracking" in state:
            label_data = state["label_tracking"]
            self._label_sampling_counts = label_data.get("label_sampling_counts", {})
            self._label_eviction_counts = label_data.get("label_eviction_counts", {})
            self._active_labels = set(label_data.get("active_labels", []))
            self._inactive_labels_fifo = label_data.get("inactive_labels_fifo", [])

        # Fix LP scores for tasks loaded from old checkpoints
        # Tasks with 0 completions should have exploration_bonus, not 0.0
        fixed_count = 0
        for task_id in self.task_tracker.get_all_tracked_tasks():
            stats = self.task_tracker.get_task_stats(task_id)
            if stats and stats["completion_count"] == 0 and stats["lp_score"] == 0.0:
                self.task_tracker.update_lp_score(task_id, self.hypers.exploration_bonus)
                fixed_count += 1

        if fixed_count > 0:
            bonus = self.hypers.exploration_bonus
            logger.info(f"LP Algorithm: Fixed {fixed_count} tasks with 0 completions to exploration_bonus={bonus}")

    def cleanup_shared_memory(self) -> None:
        """Clean up shared memory resources with better error handling."""
        try:
            self.task_tracker.cleanup_shared_memory()
        except Exception as e:
            # Log but don't raise - cleanup should be best-effort
            logging.warning(f"Failed to cleanup shared memory: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.hypers.use_shared_memory:
            self.cleanup_shared_memory()
