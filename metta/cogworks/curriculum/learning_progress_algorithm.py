"""
Learning Progress Algorithm with integrated bidirectional scoring.

Provides intelligent task selection based on bidirectional learning progress analysis,
using fast and slow exponential moving averages to detect learning opportunities.
"""

import random
from typing import Any, Dict, List, Optional

import numpy as np

from .curriculum import CurriculumAlgorithm, CurriculumAlgorithmConfig, CurriculumTask
from .lp_scorers import BasicLPScorer, BidirectionalLPScorer, LPScorer
from .stats import CacheCoordinator, LPStatsAggregator
from .task_tracker import TaskTracker


class LearningProgressConfig(CurriculumAlgorithmConfig):
    """Configuration for learning progress with bidirectional scoring as default."""

    type: str = "learning_progress"

    # Bidirectional learning progress settings (now default)
    use_bidirectional: bool = True
    use_baseline_normalization: bool = (
        True  # Normalize by baseline to get "mastery" score p_i = (TSR_i - B_i) / (1.0 - B_i)
    )
    ema_timescale: float = 0.1  # EMA learning rate (0.1 = updates in ~10 samples)
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

    # Performance and memory management
    max_slice_axes: int = 3  # Updated terminology

    # Memory backend configuration
    task_struct_size: int = 13  # Size of task data structure in shared memory (includes ema_squared)
    enable_detailed_slice_logging: bool = False  # Updated terminology
    use_shared_memory: bool = True  # Enabled by default for production use
    session_id: Optional[str] = None  # Session ID for shared memory, None = auto-generate unique

    # Logging configuration
    show_curriculum_troubleshooting_logging: bool = False  # Show high-cardinality per-task metrics for debugging

    def algorithm_type(self) -> str:
        return "learning_progress"

    def create(self, num_tasks: int) -> "LearningProgressAlgorithm":
        return LearningProgressAlgorithm(num_tasks, self)


class LearningProgressAlgorithm(CurriculumAlgorithm):
    """
    Learning Progress Algorithm with integrated bidirectional scoring.

    Uses bidirectional learning progress by default, combining fast and slow
    exponential moving averages to detect learning opportunities and guide
    intelligent task selection.
    """

    def __init__(self, num_tasks: int, hypers: LearningProgressConfig):
        super().__init__(num_tasks, hypers)

        self.num_tasks = num_tasks
        self.hypers: LearningProgressConfig = hypers

        # Initialize task tracker (unified implementation with configurable backend)
        # Note: max_memory_tasks is automatically set to num_active_tasks
        self.task_tracker = TaskTracker(
            max_memory_tasks=hypers.num_active_tasks,
            ema_alpha=hypers.task_tracker_ema_alpha,
            session_id=hypers.session_id if hypers.use_shared_memory else None,
            use_shared_memory=hypers.use_shared_memory,
            task_struct_size=hypers.task_struct_size,
            default_success_threshold=hypers.task_default_success_threshold,
            default_generator_type=hypers.task_default_generator_type,
        )

        # Note: slice_analyzer is already initialized in parent class via StatsLogger

        # Initialize scorer strategy
        self.scorer: LPScorer = BidirectionalLPScorer(hypers) if hypers.use_bidirectional else BasicLPScorer(hypers)

        # Initialize stats aggregator to centralize stats computation
        self.stats_aggregator = LPStatsAggregator(
            task_tracker=self.task_tracker,
            scorer=self.scorer,
            slice_analyzer=self.slice_analyzer,
            num_tasks=num_tasks,
        )

        # Initialize cache coordinator to centralize cache invalidation
        self.cache_coordinator = CacheCoordinator(
            stats_logger=self,
            scorer=self.scorer,
            slice_analyzer=self.slice_analyzer,
        )

        # Cache for expensive statistics computation
        self._stats_cache: Dict[str, Any] = {}
        self._stats_cache_valid = False

        # Track task labels for pool composition and sampling stats
        self._task_labels: Dict[int, str] = {}  # task_id -> label
        self._label_completion_counts: Dict[str, int] = {}  # label -> completion count
        self._label_sampling_counts: Dict[str, int] = {}  # label -> cumulative sampling count (episodes started)

        # Per-label tracking (only if troubleshooting logging enabled to prevent memory leaks)
        if hypers.show_curriculum_troubleshooting_logging:
            self._label_eviction_counts: Dict[str, int] = {}  # label -> eviction count
        else:
            self._label_eviction_counts = None

        # Per-epoch tracking (ALWAYS enabled for gini calculation)
        self._label_evictions_this_epoch: Dict[str, int] = {}  # label -> evictions this epoch
        self._label_sampling_counts_this_epoch: Dict[str, int] = {}  # label -> samples this epoch

        # Track which labels are currently active (have tasks in pool)
        self._active_labels: set[str] = set()

        # Track recently inactive labels to manage memory
        self._inactive_labels_fifo: list[str] = []  # FIFO queue of inactive labels for cleanup

    @property
    def lp_scorer(self):
        """Compatibility property for tests that expect lp_scorer attribute."""
        return self

    @property
    def exploration_bonus(self):
        """Compatibility property for tests that expect exploration_bonus attribute."""
        return self.hypers.exploration_bonus

    @property
    def _cache_valid_tasks(self):
        """Compatibility property for tests that access scorer's cache."""
        return self.scorer._cache_valid_tasks

    @property
    def _score_cache(self):
        """Compatibility property for tests that access scorer's cache."""
        return self.scorer._score_cache

    def stats(self, prefix: str = "") -> Dict[str, float]:
        """Get all statistics with optional prefix. Always includes learning progress stats."""
        cache_key = prefix if prefix else "_default"

        if self._stats_cache_valid and cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        # Get base stats (required)
        stats = self.get_base_stats()

        if self.enable_detailed_logging:
            detailed = self.get_detailed_stats()
            stats.update(detailed)

        # Add prefix to all keys
        if prefix:
            stats = {f"{prefix}{k}": v for k, v in stats.items()}

        # Cache result
        self._stats_cache[cache_key] = stats
        self._stats_cache_valid = True

        return stats

    def score_tasks(self, task_ids: List[int]) -> Dict[int, float]:
        """Score tasks using the configured method (bidirectional by default)."""
        # NEW: Use scorer strategy instead of conditionals
        return {task_id: self.scorer.score_task(task_id, self.task_tracker) for task_id in task_ids}

    def recommend_eviction(self, task_ids: List[int]) -> Optional[int]:
        """Recommend which task to evict based on learning progress."""
        if not task_ids:
            return None

        scores = self.score_tasks(task_ids)

        # Find task with minimum learning progress
        min_task_id = min(task_ids, key=lambda tid: scores.get(tid, 0.0))
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
        # Remove from task tracker (handles its own locking)
        self.task_tracker.remove_task(task_id)

        # Learning progress specific cleanup
        self._remove_task_from_scoring(task_id)

        # Remove from label tracking and clean up inactive labels
        evicted_label = self._task_labels.pop(task_id, None)
        if evicted_label:
            # Track cumulative eviction count for this label (only if troubleshooting logging enabled)
            if self._label_eviction_counts is not None:
                self._label_eviction_counts[evicted_label] = self._label_eviction_counts.get(evicted_label, 0) + 1

            # Track per-epoch eviction count (ALWAYS enabled for gini calculation)
            self._label_evictions_this_epoch[evicted_label] = self._label_evictions_this_epoch.get(evicted_label, 0) + 1

            # Check if this label still has any active tasks
            if evicted_label not in self._task_labels.values():
                # No more tasks with this label - remove from active set and track as inactive
                self._active_labels.discard(evicted_label)
                self._inactive_labels_fifo.append(evicted_label)

                # Clean up old inactive labels to prevent memory leak
                self._cleanup_old_inactive_labels()

        # Remove from slice analyzer to prevent memory leak
        self.slice_analyzer.remove_task(task_id)

        # Invalidate stats cache when task state changes
        self.cache_coordinator.invalidate_stats_cache()

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
                self._label_completion_counts.pop(old_label, None)
                self._label_sampling_counts.pop(old_label, None)

                # Clean up eviction counts if tracking them
                if self._label_eviction_counts is not None:
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
        if task_id in self._task_labels:
            label = self._task_labels[task_id]
            self._label_sampling_counts[label] = self._label_sampling_counts.get(label, 0) + 1
            self._label_sampling_counts_this_epoch[label] = self._label_sampling_counts_this_epoch.get(label, 0) + 1

    def get_and_reset_evictions_this_epoch(self) -> Dict[str, int]:
        """Get per-epoch evictions and reset the counter.

        Returns:
            Dictionary mapping label -> eviction count this epoch
        """
        evictions = self._label_evictions_this_epoch.copy()
        self._label_evictions_this_epoch.clear()
        return evictions

    def get_and_reset_sampling_counts_this_epoch(self) -> Dict[str, int]:
        """Get per-epoch sampling counts and reset the counter.

        Returns:
            Dictionary mapping label -> sampling count this epoch
        """
        sampling_counts = self._label_sampling_counts_this_epoch.copy()
        self._label_sampling_counts_this_epoch.clear()
        return sampling_counts

    def reset_epoch_counters(self) -> None:
        """Reset per-epoch counters at the start of a new epoch.

        This is called by the training infrastructure at epoch boundaries
        to ensure per-epoch metrics start fresh.
        """
        self._label_sampling_counts_this_epoch.clear()
        self._label_evictions_this_epoch.clear()

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance using the scorer strategy."""
        # NEW: Update scorer's internal state
        self.scorer.update_with_score(task_id, score)

        # Calculate RAW LP score for storage (before sigmoid/normalization)
        # This is used for Gini coefficient calculation to measure true inequality
        # in learning progress, not just the normalized sampling distribution
        if hasattr(self.scorer, "get_raw_lp_score"):
            lp_score = self.scorer.get_raw_lp_score(task_id, self.task_tracker)
        else:
            # Fallback for scorers without raw LP (e.g., BasicLPScorer)
            lp_score = self.scorer.score_task(task_id, self.task_tracker)

        # Single atomic update to task tracker with both score and LP score
        # This ensures consistency and avoids multiple writes to shared memory
        self.task_tracker.update_task_performance(task_id, score, lp_score=lp_score)

        # Track completion counts by label
        if task_id in self._task_labels:
            label = self._task_labels[task_id]
            old_count = self._label_completion_counts.get(label, 0)
            self._label_completion_counts[label] = old_count + 1

            # Debug: Removed excessive logging

        # Invalidate stats cache when task performance changes
        self.cache_coordinator.invalidate_stats_cache()

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

    def get_learning_progress_score(self, task_id: int, task_tracker=None) -> float:
        """Get learning progress score for a specific task (compatibility method for tests)."""
        # NEW: Use scorer strategy
        return self.scorer.score_task(task_id, self.task_tracker)

    def get_task_lp_score(self, task_id: int) -> float:
        """Get learning progress score for a specific task (alias for get_learning_progress_score)."""
        return self.get_learning_progress_score(task_id)

    def get_task_raw_lp_score(self, task_id: int) -> float:
        """Get raw learning progress score for a specific task (before z-score normalization)."""
        if hasattr(self.scorer, "get_raw_lp_score"):
            return self.scorer.get_raw_lp_score(task_id, self.task_tracker)
        # Fallback to regular LP score if scorer doesn't support raw LP
        return self.get_learning_progress_score(task_id)

    def get_task_postzscored_lp_score(self, task_id: int) -> float:
        """Get post-z-score learning progress score for a specific task (after z-score, before sigmoid)."""
        if hasattr(self.scorer, "get_postzscored_lp_score"):
            return self.scorer.get_postzscored_lp_score(task_id, self.task_tracker)
        # Fallback to regular LP score if scorer doesn't support post-z-score LP
        return self.get_learning_progress_score(task_id)

    def get_stats(self) -> Dict[str, float]:
        """Get learning progress statistics (compatibility method for tests)."""
        return self.scorer.get_stats()

    def update_task_with_slice_values(self, task_id: int, score: float, slice_values: Dict[str, Any]) -> None:
        """Update task performance including slice values for analysis."""
        # First update performance
        self.update_task_performance(task_id, score)

        # Then update slice analyzer
        self.slice_analyzer.update_task_completion(task_id, slice_values, score)

    def on_task_created(self, task: CurriculumTask) -> None:
        """Handle task creation by tracking it."""
        self.task_tracker.track_task_creation(task._task_id)

        # Track task label for pool composition stats
        if hasattr(task, "get_label"):
            label = task.get_label()
            if label:
                self._task_labels[task._task_id] = label

                # If label was inactive, remove it from the inactive queue (reactivating it)
                if label in self._inactive_labels_fifo:
                    self._inactive_labels_fifo.remove(label)

                self._active_labels.add(label)

        # Extract and update slice values if available
        slice_values = task.get_slice_values()
        if slice_values:
            # Initial tracking with neutral score
            self.slice_analyzer.update_task_completion(task._task_id, slice_values, 0.5)

        # Invalidate stats cache when task state changes
        self.cache_coordinator.invalidate_stats_cache()

    def get_pool_composition_stats(self) -> Dict[str, Dict[str, int]]:
        """Get pool composition and sampling statistics by label.

        Returns:
            Dictionary with 'pool_composition' and 'sampling_counts' keys,
            each containing label->count mappings.
        """
        # Count labels currently in pool
        pool_composition = {}
        for label in self._task_labels.values():
            pool_composition[label] = pool_composition.get(label, 0) + 1

        # Return per-epoch sampling counts (reset each epoch)
        return {
            "pool_composition": pool_composition,
            "sampling_counts": self._label_sampling_counts_this_epoch.copy(),
        }

    def get_base_stats(self) -> Dict[str, float]:
        """Get basic statistics that all algorithms must provide."""
        stats = self.stats_aggregator.get_base_stats()

        # Add pool composition stats (logged every epoch)
        composition_data = self.get_pool_composition_stats()

        # Add pool composition (number of each label in memory)
        for label, count in composition_data["pool_composition"].items():
            stats[f"pool_composition/{label}"] = float(count)

        # Add sampling counts (number of times each label was sampled)
        for label, count in composition_data["sampling_counts"].items():
            stats[f"sampling_counts/{label}"] = float(count)

        # Add eviction counts (number of times each label was evicted)
        if self._label_eviction_counts is not None:
            for label, count in self._label_eviction_counts.items():
                stats[f"eviction_counts/{label}"] = float(count)

        # Calculate comprehensive Gini coefficients (replaces old pool_occupancy_gini and pool_lp_gini)
        gini_stats = self._calculate_comprehensive_gini_coefficients()
        stats.update(gini_stats)

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
        import logging
        import math

        logger = logging.getLogger(__name__)

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

    def _calculate_comprehensive_gini_coefficients(self) -> Dict[str, float]:
        """Calculate Gini coefficients at each stage of the LP calculation pipeline.

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
        import logging

        logger = logging.getLogger(__name__)

        gini_stats = {}

        # Get all tracked tasks
        all_task_ids = self.task_tracker.get_all_tracked_tasks()

        # Debug: Log task tracking status
        gini_stats["debug/num_tracked_tasks"] = float(len(all_task_ids)) if all_task_ids else 0.0

        # Also check task_id_to_index mapping
        if hasattr(self.task_tracker, "_task_id_to_index"):
            gini_stats["debug/task_id_mapping_size"] = float(len(self.task_tracker._task_id_to_index))

        if not all_task_ids:
            mapping_size = (
                len(self.task_tracker._task_id_to_index) if hasattr(self.task_tracker, "_task_id_to_index") else "N/A"
            )
            logger.warning(f"No tracked tasks! task_id_to_index size: {mapping_size}")
            return gini_stats

        # Collect task-level data
        completion_counts = []
        raw_lp_scores = []
        z_scored_lp_scores = []
        sampling_probs = []
        task_labels_list = []

        num_tasks_with_stats = 0
        for task_id in all_task_ids:
            task_stats = self.task_tracker.get_task_stats(task_id)
            if task_stats:
                num_tasks_with_stats += 1
                # 1. Completion counts (for pool occupancy Gini)
                completion_count = float(task_stats["completion_count"])
                completion_counts.append(completion_count)

                # 2. Raw LP scores (stored in tracker after our fix)
                raw_lp = float(task_stats.get("lp_score", 0.0))
                raw_lp_scores.append(raw_lp)

                # 3. Z-scored LP scores (if bidirectional scorer available)
                if hasattr(self.scorer, "get_postzscored_lp_score"):
                    z_scored = self.scorer.get_postzscored_lp_score(task_id, self.task_tracker)
                    z_scored_lp_scores.append(float(z_scored))

                # 4. Final sampling probabilities
                sampling_prob = self.scorer.score_task(task_id, self.task_tracker)
                sampling_probs.append(float(sampling_prob))

                # Track label for aggregations
                if task_id in self._task_labels:
                    task_labels_list.append(self._task_labels[task_id])
                else:
                    task_labels_list.append("unknown")

                # Log per-task metrics if troubleshooting enabled
                if self.hypers.show_curriculum_troubleshooting_logging:
                    gini_stats[f"task_metrics/{task_id}/completion_count"] = completion_count
                    gini_stats[f"task_metrics/{task_id}/raw_lp"] = raw_lp
                    gini_stats[f"task_metrics/{task_id}/sampling_prob"] = sampling_prob
            else:
                logger.warning(f"No stats found for task {task_id}")

        # Debug: Log collection results
        gini_stats["debug/num_tasks_with_stats"] = float(num_tasks_with_stats)

        # === 1. Pool Occupancy Gini (task-level completion counts) ===
        if completion_counts:
            gini_stats["curriculum_gini/pool_occupancy"] = self._calculate_gini_coefficient(completion_counts)
        else:
            logger.warning("No completion counts collected for Gini calculation!")

        # === 2. Raw LP Scores Gini (task-level) ===
        if raw_lp_scores:
            # Debug: Log LP score distribution to diagnose Gini=0
            gini_stats["curriculum_gini/raw_lp_scores"] = self._calculate_gini_coefficient(raw_lp_scores)
            # Add diagnostic stats
            gini_stats["debug/raw_lp_min"] = float(min(raw_lp_scores))
            gini_stats["debug/raw_lp_max"] = float(max(raw_lp_scores))
            gini_stats["debug/raw_lp_mean"] = float(sum(raw_lp_scores) / len(raw_lp_scores))
            gini_stats["debug/raw_lp_nonzero_count"] = float(sum(1 for x in raw_lp_scores if x != 0))
            gini_stats["debug/raw_lp_total_count"] = float(len(raw_lp_scores))
            gini_stats["debug/raw_lp_unique_count"] = float(len(set(raw_lp_scores)))
            gini_stats["debug/raw_lp_std"] = float(np.std(raw_lp_scores)) if len(raw_lp_scores) > 1 else 0.0

            # Log warning if all scores are identical (causes Gini=0)
            if len(set(raw_lp_scores)) == 1:
                logger.warning(
                    f"All {len(raw_lp_scores)} raw LP scores are identical ({raw_lp_scores[0]:.6f}), "
                    "Gini will be 0. This likely means tasks haven't been completed enough yet."
                )

        # === 3. Raw LP Scores by Label (aggregated) ===
        if raw_lp_scores and task_labels_list:
            label_lp_sums = {}
            for label, lp in zip(task_labels_list, raw_lp_scores, strict=True):
                label_lp_sums[label] = label_lp_sums.get(label, 0.0) + lp

            if label_lp_sums:
                label_lp_values = list(label_lp_sums.values())
                gini_stats["curriculum_gini/raw_lp_by_label"] = self._calculate_gini_coefficient(label_lp_values)

        # === 4. Z-Scored LP Scores Gini (task-level) ===
        if z_scored_lp_scores:
            gini_stats["curriculum_gini/zscored_lp_scores"] = self._calculate_gini_coefficient(
                [abs(z) for z in z_scored_lp_scores]  # Use absolute values for Gini
            )

        # === 5. Final Sampling Probabilities Gini (task-level) ===
        if sampling_probs:
            gini_stats["curriculum_gini/sampling_probs"] = self._calculate_gini_coefficient(sampling_probs)

        # === 5b. Sampling Probabilities by Label (aggregated) ===
        if sampling_probs and task_labels_list:
            label_prob_sums = {}
            for label, prob in zip(task_labels_list, sampling_probs, strict=True):
                label_prob_sums[label] = label_prob_sums.get(label, 0.0) + prob

            if label_prob_sums:
                label_prob_values = list(label_prob_sums.values())
                gini_stats["curriculum_gini/sampling_probs_by_label"] = self._calculate_gini_coefficient(
                    label_prob_values
                )

        # === 6. Sampling Counts by Label (per-epoch) ===
        if self._label_sampling_counts_this_epoch:
            label_sampling_values = list(self._label_sampling_counts_this_epoch.values())
            gini_stats["curriculum_gini/sampling_by_label"] = self._calculate_gini_coefficient(label_sampling_values)

        # === 7. Eviction Counts by Label ===
        if self._label_eviction_counts:
            label_eviction_values = list(self._label_eviction_counts.values())
            if label_eviction_values and sum(label_eviction_values) > 0:
                gini_stats["curriculum_gini/evictions_by_label"] = self._calculate_gini_coefficient(
                    label_eviction_values
                )

        # === 8. Pool Composition by Label ===
        composition_data = self.get_pool_composition_stats()
        if composition_data["pool_composition"]:
            pool_comp_values = list(composition_data["pool_composition"].values())
            gini_stats["curriculum_gini/pool_composition_by_label"] = self._calculate_gini_coefficient(pool_comp_values)

        # Calculate "selectivity loss" metrics (how much Gini decreases at each stage)
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
        return self.stats_aggregator.get_detailed_stats()

    def get_state(self) -> Dict[str, Any]:
        """Get learning progress algorithm state for checkpointing."""
        return {
            "type": self.hypers.algorithm_type(),
            "hypers": self.hypers.model_dump(),
            "task_tracker": self.task_tracker.get_state(),
            "scorer": self.scorer.get_state(),
            "label_tracking": {
                "task_labels": self._task_labels,
                "label_completion_counts": self._label_completion_counts,
                "label_sampling_counts": self._label_sampling_counts,
                "label_eviction_counts": self._label_eviction_counts,
                "active_labels": list(self._active_labels),
                "inactive_labels_fifo": self._inactive_labels_fifo,
            },
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load learning progress algorithm state from checkpoint."""
        # Restore task tracker
        self.task_tracker.load_state(state["task_tracker"])

        # Restore scorer state
        if "scorer" in state:
            self.scorer.load_state(state["scorer"])

        # Restore label tracking state (if available, for backward compatibility)
        if "label_tracking" in state:
            label_data = state["label_tracking"]
            self._task_labels = label_data.get("task_labels", {})
            self._label_completion_counts = label_data.get("label_completion_counts", {})
            self._label_sampling_counts = label_data.get("label_sampling_counts", {})

            # Handle eviction counts (may be None if troubleshooting disabled)
            eviction_counts = label_data.get("label_eviction_counts")
            if self._label_eviction_counts is not None and eviction_counts is not None:
                self._label_eviction_counts = eviction_counts

            self._active_labels = set(label_data.get("active_labels", []))
            self._inactive_labels_fifo = label_data.get("inactive_labels_fifo", [])

    def get_per_label_lp_scores(self, task_pool: dict) -> dict[str, dict[str, float]]:
        """Get per-label LP scores for troubleshooting.

        Args:
            task_pool: Dictionary mapping task_id -> CurriculumTask from the curriculum

        Returns:
            Dict mapping label -> {raw, postzscored, prob}
            - raw: average raw LP score for tasks in this label
            - postzscored: average post-z-scored LP for tasks in this label
            - prob: total sampling probability for this label (sum of task probs)
        """
        if not self.hypers.show_curriculum_troubleshooting_logging:
            return {}

        per_label: dict[str, dict[str, float]] = {}
        for task_id, task in task_pool.items():
            label = task.get_label()
            if not label or not isinstance(label, str):
                continue

            if label not in per_label:
                per_label[label] = {"raw": 0.0, "postzscored": 0.0, "prob": 0.0, "count": 0}

            # Accumulate scores
            # - raw and postzscored are averaged (diagnostic metrics)
            # - prob is summed (represents total sampling probability for this label)
            per_label[label]["raw"] += self.get_task_raw_lp_score(task_id)
            per_label[label]["postzscored"] += self.get_task_postzscored_lp_score(task_id)
            per_label[label]["prob"] += self.get_task_lp_score(task_id)  # Sum for sampling probability
            per_label[label]["count"] += 1

        # Process scores per label
        # First, calculate total sum of all LP scores for normalization
        total_prob_sum = sum(scores["prob"] for scores in per_label.values())

        for _label, scores in per_label.items():
            count = scores.pop("count")
            if count > 0:
                # Average raw and postzscored (diagnostic metrics)
                scores["raw"] /= count
                scores["postzscored"] /= count
                # Normalize prob to get true sampling probability
                # (sum of label probs / total sum = probability of sampling this label)
                if total_prob_sum > 0:
                    scores["prob"] /= total_prob_sum
                else:
                    scores["prob"] = 0.0

        return per_label

    def cleanup_shared_memory(self) -> None:
        """Clean up shared memory resources with better error handling."""
        if not hasattr(self, "task_tracker"):
            return

        try:
            # TaskTracker always has cleanup_shared_memory method
            self.task_tracker.cleanup_shared_memory()
        except Exception as e:
            # Log but don't raise - cleanup should be best-effort
            import logging

            logging.warning(f"Failed to cleanup shared memory: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, "hypers") and getattr(self.hypers, "use_shared_memory", False):
            self.cleanup_shared_memory()
