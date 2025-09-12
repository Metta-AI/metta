"""Learning Progress Algorithm implementation using modular components.

This module provides the main LearningProgressAlgorithm that orchestrates
specialized components for high-performance curriculum learning.
"""

import logging
from typing import Dict, List, Optional

from .curriculum import CurriculumAlgorithm, CurriculumAlgorithmConfig, CurriculumTask
from .learning_progress_modules import (
    BucketAnalyzer,
    LearningProgressScorer,
    TaskTracker,
)

logger = logging.getLogger(__name__)


class LearningProgressConfig(CurriculumAlgorithmConfig):
    """Configuration for the learning progress algorithm."""

    type: str = "learning_progress"

    # Core algorithm parameters
    ema_timescale: float = 0.001
    exploration_bonus: float = 0.1

    # Performance and memory management
    max_memory_tasks: int = 1000
    max_bucket_axes: int = 3
    logging_detailed_slices: bool = False  # Disabled by default for performance

    # Bidirectional learning progress parameters
    use_bidirectional: bool = True  # Default to True for better performance
    progress_smoothing: float = 0.05
    num_active_tasks: int = 16
    rand_task_rate: float = 0.25
    sample_threshold: int = 10
    memory: int = 25

    def algorithm_type(self) -> str:
        return "learning_progress"

    def create(self, num_tasks: int) -> "LearningProgressAlgorithm":
        return LearningProgressAlgorithm(num_tasks, self)


class LearningProgressAlgorithm(CurriculumAlgorithm):
    """
    Learning Progress Algorithm with integrated scoring functionality.

    Combines task tracking, learning progress scoring, and bucket analysis
    into a cohesive algorithm for intelligent task selection based on
    performance variance and exploration needs.
    """

    def __init__(self, num_tasks: int, hypers: LearningProgressConfig):
        # Initialize parent class if it has an __init__ method
        if hasattr(super(), "__init__"):
            super().__init__(num_tasks, hypers)

        self.num_tasks = num_tasks
        self.hypers = hypers

        # Create modular components
        self.task_tracker = TaskTracker(
            max_memory_tasks=hypers.max_memory_tasks,
            max_bucket_axes=hypers.max_bucket_axes,
            logging_detailed_slices=hypers.logging_detailed_slices,
        )

        # Create unified learning progress scorer
        scorer_mode = "bidirectional" if hypers.use_bidirectional else "standard"
        self.lp_scorer = LearningProgressScorer(
            mode=scorer_mode,
            ema_timescale=hypers.ema_timescale,
            exploration_bonus=hypers.exploration_bonus,
            progress_smoothing=hypers.progress_smoothing,
            num_active_tasks=hypers.num_active_tasks,
            rand_task_rate=hypers.rand_task_rate,
            sample_threshold=hypers.sample_threshold,
            memory=hypers.memory,
        )

        # Create bucket analyzer
        self.bucket_analyzer = BucketAnalyzer(
            max_bucket_axes=hypers.max_bucket_axes,
            logging_detailed_slices=hypers.logging_detailed_slices,
        )

        # Cache for expensive stats computation
        self._stats_cache = {}
        self._stats_cache_valid = False

        # Curriculum reference for accessing RNG
        self._curriculum = None

    def set_curriculum_reference(self, curriculum) -> None:
        """Set reference to curriculum for accessing its RNG."""
        self._curriculum = curriculum

    # CurriculumAlgorithm interface implementation

    def score_tasks(self, task_ids: List[int]) -> Dict[int, float]:
        """Score tasks for selection based on learning progress."""
        return self.lp_scorer.score_tasks(task_ids)

    def recommend_eviction(self, task_ids: List[int]) -> Optional[int]:
        """Recommend task to evict based on learning progress."""
        if not task_ids:
            return None

        scores = self.score_tasks(task_ids)

        # Find task with minimum learning progress
        min_task_id = min(task_ids, key=lambda tid: scores.get(tid, 0.0))
        return min_task_id

    def should_evict_task(self, task_id: int, min_presentations: int = 5) -> bool:
        """Check if a task should be evicted based on criteria.

        Args:
            task_id: The task to check
            min_presentations: Minimum number of task presentations before eviction

        Returns:
            True if task should be evicted (enough presentations + low learning progress)
        """
        # Check minimum presentations requirement
        task_stats = self.task_tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < min_presentations:
            return False

        # Check if this task has low learning progress compared to others
        all_task_ids = self.task_tracker.get_tracked_task_ids()
        if len(all_task_ids) <= 1:
            return False

        scores = self.score_tasks(all_task_ids)
        task_score = scores.get(task_id, 0.0)

        # Evict if this task is in the bottom 20% of learning progress scores
        sorted_scores = sorted(scores.values())
        threshold_index = max(0, int(len(sorted_scores) * 0.2))
        threshold_score = sorted_scores[threshold_index] if sorted_scores else 0.0

        return task_score <= threshold_score

    def on_task_evicted(self, task_id: int) -> None:
        """Clean up when a task is evicted."""
        # Get bucket values before removing task
        task_stats = self.task_tracker.get_task_stats(task_id)
        bucket_values = task_stats.get("bucket_values", {})

        # Remove from all components
        self.task_tracker.remove_task(task_id)
        self.lp_scorer.remove_task(task_id)
        self.bucket_analyzer.remove_task_bucket_data(task_id, bucket_values)

        # Invalidate stats cache when task state changes
        self._stats_cache_valid = False

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance across all components."""
        # Update task tracker
        self.task_tracker.update_task_performance(task_id, score)

        # Update learning progress scorer
        self.lp_scorer.update_task_ema(task_id, score)

        # Update bucket analyzer
        task_stats = self.task_tracker.get_task_stats(task_id)
        bucket_values = task_stats.get("bucket_values", {})
        if bucket_values:
            self.bucket_analyzer.update_bucket_completions(task_id, bucket_values)

        # Invalidate stats cache when performance updates
        self._stats_cache_valid = False

    def on_task_created(self, task: CurriculumTask) -> None:
        """Handle new task creation."""
        task_id = task._task_id
        bucket_values = task.get_bucket_values()

        # Track task creation in all components
        self.task_tracker.track_task_creation(task_id, bucket_values)

        # Invalidate stats cache when new tasks are created
        self._stats_cache_valid = False

    def stats(self, prefix: str = "") -> Dict[str, float]:
        """Get comprehensive statistics from all components."""
        # Use cached stats if valid to avoid expensive recomputation
        cache_key = f"stats_{prefix}"
        if self._stats_cache_valid and cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        stats = {}

        # Add prefix helper
        def add_prefix(d: Dict[str, float], p: str) -> Dict[str, float]:
            return {f"{prefix}{p}{k}": v for k, v in d.items()}

        # Task tracker stats
        task_stats = {
            "total_tracked_tasks": float(len(self.task_tracker.get_tracked_task_ids())),
            "total_completions": float(self.task_tracker.get_total_completions()),
        }
        stats.update(add_prefix(task_stats, "tracker/"))

        # Learning progress scorer stats
        if self.lp_scorer.mode == "bidirectional":
            lp_stats = self.lp_scorer.get_bidirectional_stats()
            # Convert numpy arrays to scalar values for logging
            for key, value in lp_stats.items():
                if value is not None:
                    if hasattr(value, "mean"):
                        stats[f"{prefix}lp/{key}"] = float(value.mean())
                    elif hasattr(value, "__len__") and len(value) > 0:
                        stats[f"{prefix}lp/{key}_count"] = float(len(value))
                    else:
                        stats[f"{prefix}lp/{key}"] = float(value)

        # Add basic scorer stats using get_stats method
        basic_lp_stats = self.lp_scorer.get_stats()
        stats.update(add_prefix(basic_lp_stats, "lp/"))

        # Basic stats for any scorer
        stats[f"{prefix}lp/scorer_mode_id"] = 1.0 if self.lp_scorer.mode == "bidirectional" else 0.0

        # Bucket analyzer stats
        bucket_stats_raw = self.bucket_analyzer.get_bucket_stats()
        bucket_stats = {k: float(v) for k, v in bucket_stats_raw.items() if isinstance(v, (int, float))}
        stats.update(add_prefix(bucket_stats, "bucket/"))

        # Cache the result
        self._stats_cache[cache_key] = stats
        self._stats_cache_valid = True

        return stats

    def _choose_task_from_list(self, task_ids: List[int]) -> int:
        """Choose a task from a specific list of task IDs using learning progress scores."""
        if not task_ids:
            raise ValueError("No tasks provided to sample from")

        scores = self.score_tasks(task_ids)

        # Convert scores to probabilities for sampling
        score_values = [scores.get(task_id, 0.0) for task_id in task_ids]
        total_score = sum(score_values)

        if total_score > 0:
            probabilities = [score / total_score for score in score_values]
            # Use curriculum's RNG for deterministic behavior
            if self._curriculum is not None:
                return self._curriculum._rng.choices(task_ids, weights=probabilities)[0]
            else:
                # Fallback to numpy for backwards compatibility
                import numpy as np

                return np.random.choice(task_ids, p=probabilities)
        else:
            # Use curriculum's RNG for deterministic behavior
            if self._curriculum is not None:
                return self._curriculum._rng.choice(task_ids)
            else:
                # Fallback to random for backwards compatibility
                import random

                return random.choice(task_ids)
