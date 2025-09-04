"""
Learning Progress Algorithm using composition of focused components.

This algorithm uses smaller, focused components that each handle a single responsibility.
"""

from typing import Dict, List, Optional

from .curriculum import CurriculumAlgorithm, CurriculumAlgorithmConfig, CurriculumTask
from .learning_progress_modules import BucketAnalyzer, LearningProgressScorer, TaskTracker


class LearningProgressConfig(CurriculumAlgorithmConfig):
    """Configuration for the learning progress algorithm."""

    type: str = "learning_progress"

    # Core algorithm parameters
    ema_timescale: float = 0.001
    exploration_bonus: float = 0.1

    # Performance and memory management
    max_memory_tasks: int = 1000
    max_bucket_axes: int = 3
    enable_detailed_bucket_logging: bool = False  # Disabled by default for performance

    def algorithm_type(self) -> str:
        return "learning_progress"

    def create(self, num_tasks: int) -> "LearningProgressAlgorithm":
        return LearningProgressAlgorithm(num_tasks, self)


class LearningProgressAlgorithm(CurriculumAlgorithm):
    """
    Learning Progress Algorithm using focused components.

    This version demonstrates clean separation of concerns:
    - TaskTracker: Handles task memory and performance history
    - LearningProgressScorer: Calculates learning progress scores
    - BucketAnalyzer: Tracks completion patterns across parameter space

    The algorithm coordinates these components but doesn't mix responsibilities.
    """

    def __init__(self, num_tasks: int, hypers: LearningProgressConfig):
        super().__init__(num_tasks, hypers, initialize_weights=False)

        self.hypers = hypers

        # Initialize focused components
        self.task_tracker = TaskTracker(max_memory_tasks=hypers.max_memory_tasks)
        self.lp_scorer = LearningProgressScorer(
            ema_timescale=hypers.ema_timescale, exploration_bonus=hypers.exploration_bonus
        )
        self.bucket_analyzer = BucketAnalyzer(
            max_bucket_axes=hypers.max_bucket_axes, enable_detailed_logging=hypers.enable_detailed_bucket_logging
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
        return self.lp_scorer.score_tasks(task_ids, self.task_tracker)

    def recommend_eviction(self, task_ids: List[int]) -> Optional[int]:
        """Recommend task to evict based on learning progress."""
        return self.lp_scorer.recommend_eviction(task_ids, self.task_tracker)

    def on_task_evicted(self, task_id: int) -> None:
        """Clean up when a task is evicted."""
        self.task_tracker.remove_task(task_id)
        self.lp_scorer.remove_task(task_id)
        self.bucket_analyzer.remove_task(task_id)

        # Invalidate stats cache when task state changes
        self._stats_cache_valid = False

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance across all components."""
        # Update task tracking
        self.task_tracker.update_task_performance(task_id, score)

        # Update learning progress EMA
        self.lp_scorer.update_task_ema(task_id, score)

        # Invalidate stats cache when performance updates
        self._stats_cache_valid = False

    def on_task_created(self, task: CurriculumTask) -> None:
        """Handle new task creation."""
        task_id = task._task_id

        # Track the new task
        self.task_tracker.track_task_creation(task_id)

        # Extract and track bucket values
        bucket_values = self.bucket_analyzer.extract_bucket_values(task)
        if bucket_values:
            # Initialize bucket tracking with default score
            self.bucket_analyzer.track_task_completion(task_id, bucket_values, 0.0)

        # Invalidate stats cache when new tasks are created
        self._stats_cache_valid = False

    def stats(self, prefix: str = "") -> Dict[str, float]:
        """Get comprehensive statistics from all components."""
        # Use cached stats if valid to avoid expensive recomputation
        if self._stats_cache_valid and prefix in self._stats_cache:
            return self._stats_cache[prefix]

        stats = {}

        # Add prefix to all keys
        def add_prefix(d: Dict[str, float], p: str) -> Dict[str, float]:
            return {f"{prefix}{p}{k}": v for k, v in d.items()}

        # Task tracking stats
        stats.update(add_prefix(self.task_tracker.get_global_stats(), "tracker/"))

        # Learning progress stats
        stats.update(add_prefix(self.lp_scorer.get_stats(), "lp/"))

        # Bucket analysis stats
        stats.update(add_prefix(self.bucket_analyzer.get_global_stats(), "buckets/"))

        # Detailed bucket density stats (if enabled) - this is expensive
        if self.hypers.enable_detailed_bucket_logging:
            density_stats = self.bucket_analyzer.get_completion_density_stats()
            for bucket_name, bucket_stats in density_stats.items():
                bucket_prefix = f"bucket_{bucket_name}/"
                stats.update(add_prefix(bucket_stats, bucket_prefix))

        # Cache the result
        self._stats_cache[prefix] = stats
        self._stats_cache_valid = True

        return stats

    def _choose_task(self) -> int:
        """
        Choose a task using learning progress guided sampling.

        This delegates to the parent class implementation which uses
        score_tasks() to get weights for weighted sampling.
        """
        # Use parent implementation that calls our score_tasks method
        return super()._choose_task()

    def _choose_task_from_list(self, task_ids: List[int]) -> int:
        """Choose a task from a specific list of task IDs using learning progress scores."""
        if not task_ids:
            raise ValueError("No tasks provided to sample from")

        # Ensure all tasks are tracked
        for task_id in task_ids:
            if task_id not in self.task_tracker._task_memory:
                self.task_tracker.track_task_creation(task_id)

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
