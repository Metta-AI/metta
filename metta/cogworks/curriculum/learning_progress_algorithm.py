"""
Learning Progress Algorithm using composition of focused components.

This algorithm uses smaller, focused components that each handle a single responsibility.
"""

from typing import Dict, List, Optional

import numpy as np

from .curriculum import CurriculumAlgorithm, CurriculumAlgorithmConfig, CurriculumTask, TaskTracker
from .stats import BucketAnalyzer


class LearningProgressScorer:
    """Calculates learning progress scores for tasks based on performance trends."""

    def __init__(self, ema_timescale: float = 0.001, exploration_bonus: float = 0.1):
        self.ema_timescale = ema_timescale
        self.exploration_bonus = exploration_bonus

        # EMA tracking for each task: task_id -> (ema_score, ema_squared, num_samples)
        self._task_emas: Dict[int, tuple[float, float, int]] = {}

        # Cache for learning progress scores to avoid recomputation
        self._score_cache: Dict[int, float] = {}
        self._cache_valid_tasks: set[int] = set()

    def update_task_ema(self, task_id: int, score: float) -> None:
        """Update EMA tracking for a task with new score."""
        if task_id not in self._task_emas:
            self._task_emas[task_id] = (score, score * score, 1)
        else:
            ema_score, ema_squared, num_samples = self._task_emas[task_id]

            # Update EMAs
            alpha = min(1.0, self.ema_timescale * num_samples)
            new_ema_score = (1 - alpha) * ema_score + alpha * score
            new_ema_squared = (1 - alpha) * ema_squared + alpha * (score * score)

            self._task_emas[task_id] = (new_ema_score, new_ema_squared, num_samples + 1)

        # Invalidate cache for this task when EMA is updated
        self._cache_valid_tasks.discard(task_id)

    def get_learning_progress_score(self, task_id: int, task_tracker: TaskTracker) -> float:
        """Calculate learning progress score for a task."""
        # Return cached score if valid
        if task_id in self._cache_valid_tasks and task_id in self._score_cache:
            return self._score_cache[task_id]

        task_stats = task_tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < 2:
            # New tasks get exploration bonus
            score = self.exploration_bonus
        elif task_id not in self._task_emas:
            # Tasks without EMA tracking get exploration bonus (they're new to the scorer)
            score = self.exploration_bonus
        else:
            ema_score, ema_squared, num_samples = self._task_emas[task_id]

            # Calculate variance from EMA
            variance = max(0.0, ema_squared - ema_score * ema_score)
            std_dev = np.sqrt(variance)

            # Learning progress is approximated by variance in performance
            # High variance = actively learning, low variance = plateaued
            learning_progress = std_dev

            # Add exploration bonus for tasks with few samples
            if num_samples < 10:
                learning_progress += self.exploration_bonus * (10 - num_samples) / 10

            score = learning_progress

        # Cache the computed score
        self._score_cache[task_id] = score
        self._cache_valid_tasks.add(task_id)
        return score

    def score_tasks(self, task_ids: List[int], task_tracker: TaskTracker) -> Dict[int, float]:
        """Score all provided tasks for selection probability."""
        scores = {}

        for task_id in task_ids:
            scores[task_id] = self.get_learning_progress_score(task_id, task_tracker)

        return scores

    def recommend_eviction(self, task_ids: List[int], task_tracker: TaskTracker) -> Optional[int]:
        """Recommend which task to evict based on learning progress."""
        if not task_ids:
            return None

        scores = self.score_tasks(task_ids, task_tracker)

        # Find task with minimum learning progress
        min_task_id = min(task_ids, key=lambda tid: scores.get(tid, 0.0))
        return min_task_id

    def remove_task(self, task_id: int) -> None:
        """Remove task from tracking and clear its cache."""
        self._task_emas.pop(task_id, None)
        self._score_cache.pop(task_id, None)
        self._cache_valid_tasks.discard(task_id)

    def get_stats(self) -> Dict[str, float]:
        """Get statistics about learning progress scoring."""
        if not self._task_emas:
            return {
                "num_tracked_tasks": 0,
                "mean_num_samples": 0.0,
                "mean_ema_score": 0.0,
                "mean_learning_progress": 0.0,
            }

        num_samples_list = [num_samples for _, _, num_samples in self._task_emas.values()]
        ema_scores = [ema_score for ema_score, _, _ in self._task_emas.values()]

        # Calculate mean learning progress from EMA data
        learning_progress_scores = []
        for ema_score, ema_squared, _num_samples in self._task_emas.values():
            variance = max(0.0, ema_squared - ema_score * ema_score)
            std_dev = np.sqrt(variance)
            learning_progress_scores.append(std_dev)

        return {
            "num_tracked_tasks": len(self._task_emas),
            "mean_num_samples": np.mean(num_samples_list),
            "mean_ema_score": np.mean(ema_scores),
            "mean_learning_progress": np.mean(learning_progress_scores) if learning_progress_scores else 0.0,
        }


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

        # Override the default task tracker with learning progress parameters
        self.task_tracker = TaskTracker(max_memory_tasks=hypers.max_memory_tasks)

        # Initialize focused components
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

    def should_evict_task(self, task_id: int, min_presentations: int = 5) -> bool:
        """Check if a task should be evicted based on criteria.

        Args:
            task_id: The task to check
            min_presentations: Minimum number of task presentations before eviction

        Returns:
            True if task should be evicted (enough presentations + low learning progress)
        """
        # First check basic criteria using parent implementation
        if not super().should_evict_task(task_id, min_presentations):
            return False

        # Check if this task has low learning progress compared to others
        all_task_ids = self.task_tracker.get_all_tracked_tasks()
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
        # Call parent implementation (handles task_tracker cleanup)
        super().on_task_evicted(task_id)

        # Learning progress specific cleanup
        self.lp_scorer.remove_task(task_id)
        self.bucket_analyzer.remove_task(task_id)

        # Invalidate stats cache when task state changes
        self._stats_cache_valid = False

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance across all components."""
        # Call parent implementation (handles task_tracker update)
        super().update_task_performance(task_id, score)

        # Update learning progress EMA
        self.lp_scorer.update_task_ema(task_id, score)

        # Invalidate stats cache when performance updates
        self._stats_cache_valid = False

    def on_task_created(self, task: CurriculumTask) -> None:
        """Handle new task creation."""
        # Call parent implementation (handles task_tracker)
        super().on_task_created(task)

        task_id = task._task_id

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

        # Start with parent stats (includes task tracker)
        stats = super().stats(prefix)

        # Add prefix to all keys
        def add_prefix(d: Dict[str, float], p: str) -> Dict[str, float]:
            return {f"{prefix}{p}{k}": v for k, v in d.items()}

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
