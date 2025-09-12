"""Learning Progress Algorithm implementation using modular components.

This module provides the main LearningProgressAlgorithm that orchestrates
specialized components for high-performance curriculum learning.
"""

import logging
import random
from typing import Dict, List, Optional

import numpy as np

from .curriculum import CurriculumAlgorithm, CurriculumAlgorithmConfig, CurriculumTask
from .stats import BucketAnalyzer
from .task_tracker import TaskTracker

logger = logging.getLogger(__name__)

# Constants for bidirectional learning progress
DEFAULT_SUCCESS_RATE = 0.0
DEFAULT_WEIGHT = 1.0
RANDOM_BASELINE_CAP = 0.75


class LearningProgressScorer:
    """Unified learning progress scorer supporting both standard and bidirectional modes.

    Standard mode: EMA-based variance calculation for learning progress
    Bidirectional mode: Fast/slow EMA differences with sigmoid normalization

    Key features:
    - Performance optimization: Score caching with validity tracking
    - Exploration bonus for new/under-sampled tasks
    - Mode-specific algorithms with shared interface
    """

    def __init__(
        self,
        mode: str = "standard",  # "standard" or "bidirectional"
        ema_timescale: float = 0.001,
        exploration_bonus: float = 0.1,
        # Bidirectional-specific parameters (ignored in standard mode)
        progress_smoothing: float = 0.05,
        num_active_tasks: int = 16,
        rand_task_rate: float = 0.25,
        sample_threshold: int = 10,
        memory: int = 25,
    ):
        self.mode = mode
        self.ema_timescale = ema_timescale
        self.exploration_bonus = exploration_bonus

        # Bidirectional-specific parameters
        self.progress_smoothing = progress_smoothing
        self.num_active_tasks = num_active_tasks
        self.rand_task_rate = rand_task_rate
        self.sample_threshold = sample_threshold
        self.memory = memory

        # Shared cache for learning progress scores
        self._score_cache: Dict[int, float] = {}
        self._cache_valid_tasks: set[int] = set()

        if mode == "bidirectional":
            self._init_bidirectional()
        else:
            self._init_standard()

    def _init_standard(self) -> None:
        """Initialize standard EMA-based tracking."""
        # EMA tracking for each task: task_id -> (ema_score, ema_squared, num_samples)
        self._task_emas: Dict[int, tuple[float, float, int]] = {}

    def _init_bidirectional(self) -> None:
        """Initialize bidirectional learning progress tracking."""
        # Bidirectional learning progress tracking
        self._outcomes: Dict[int, List[float]] = {}
        self._p_fast: Optional[np.ndarray] = None
        self._p_slow: Optional[np.ndarray] = None
        self._p_true: Optional[np.ndarray] = None
        self._random_baseline: Optional[np.ndarray] = None
        self._task_success_rate: np.ndarray = np.array([])
        self._counter: Dict[int, int] = {}
        self._update_mask: np.ndarray = np.array([])
        self._sample_levels: np.ndarray = np.array([])

        # Cache for task distribution and scores
        self._task_dist: Optional[np.ndarray] = None
        self._stale_dist = True

    def update_task_ema(self, task_id: int, score: float) -> None:
        """Update EMA tracking for a task with new score."""
        if self.mode == "bidirectional":
            self._update_bidirectional_ema(task_id, score)
        else:
            self._update_standard_ema(task_id, score)

    def _update_standard_ema(self, task_id: int, score: float) -> None:
        """Update standard EMA tracking for a task with new score."""
        if task_id in self._task_emas:
            ema_score, ema_squared, num_samples = self._task_emas[task_id]

            # Update EMAs
            ema_score = ema_score * (1 - self.ema_timescale) + score * self.ema_timescale
            ema_squared = ema_squared * (1 - self.ema_timescale) + (score**2) * self.ema_timescale
            num_samples += 1
        else:
            # Initialize EMAs for new task
            ema_score = score
            ema_squared = score**2
            num_samples = 1

        self._task_emas[task_id] = (ema_score, ema_squared, num_samples)

        # Invalidate cached score for this task
        self._cache_valid_tasks.discard(task_id)

    def _update_bidirectional_ema(self, task_id: int, score: float) -> None:
        """Update bidirectional EMA tracking for a task with new score."""
        # Track outcome history
        if task_id not in self._outcomes:
            self._outcomes[task_id] = []
            self._counter[task_id] = 0

        self._outcomes[task_id].append(score)
        self._counter[task_id] += 1

        # Keep memory manageable
        if len(self._outcomes[task_id]) > self.memory:
            self._outcomes[task_id] = self._outcomes[task_id][-self.memory :]

        # Invalidate caches
        self._stale_dist = True
        self._cache_valid_tasks.discard(task_id)

    def get_learning_progress_score(self, task_id: int, task_tracker=None) -> float:
        """Get learning progress score for a task (cached for performance).

        Args:
            task_id: The task ID to score
            task_tracker: Optional task tracker (for backward compatibility, ignored)
        """
        if task_id in self._cache_valid_tasks:
            return self._score_cache[task_id]

        # Calculate score based on mode
        if self.mode == "bidirectional":
            score = self._get_bidirectional_score(task_id)
        else:
            score = self._get_standard_score(task_id)

        # Cache the result
        self._score_cache[task_id] = score
        self._cache_valid_tasks.add(task_id)

        return score

    def _get_standard_score(self, task_id: int) -> float:
        """Calculate standard variance-based learning progress score."""
        # Calculate score
        if task_id not in self._task_emas:
            # New task gets exploration bonus
            return self.exploration_bonus

        ema_score, ema_squared, num_samples = self._task_emas[task_id]

        # Calculate variance as learning progress measure
        variance = max(0.0, ema_squared - ema_score**2)

        # Add exploration bonus for under-sampled tasks
        if num_samples < 10:
            exploration_factor = (10 - num_samples) / 10.0
            score = variance + self.exploration_bonus * exploration_factor
        else:
            score = variance

        return score

    def _get_bidirectional_score(self, task_id: int) -> float:
        """Calculate bidirectional learning progress score."""
        # Ensure task distribution is up to date
        if self._stale_dist:
            self._calculate_task_distribution()

        # Calculate score
        if task_id not in self._outcomes or self._counter[task_id] < self.sample_threshold:
            # New or under-sampled task gets exploration bonus
            return self.exploration_bonus

        # Get task index for accessing arrays
        task_ids = list(self._outcomes.keys())
        try:
            task_idx = task_ids.index(task_id)

            if self._task_dist is not None and task_idx < len(self._task_dist):
                return float(self._task_dist[task_idx])
            else:
                return self.exploration_bonus
        except (ValueError, IndexError):
            return self.exploration_bonus

    def score_tasks(self, task_ids: List[int]) -> Dict[int, float]:
        """Score multiple tasks for selection purposes."""
        return {task_id: self.get_learning_progress_score(task_id) for task_id in task_ids}

    def remove_task(self, task_id: int) -> None:
        """Remove a task from tracking."""
        if self.mode == "bidirectional":
            self._outcomes.pop(task_id, None)
            self._counter.pop(task_id, None)
            self._stale_dist = True
        else:
            self._task_emas.pop(task_id, None)

        # Clear from shared cache
        self._score_cache.pop(task_id, None)
        self._cache_valid_tasks.discard(task_id)

    def clear_cache(self) -> None:
        """Clear the score cache (forces recomputation)."""
        self._score_cache.clear()
        self._cache_valid_tasks.clear()

        if self.mode == "bidirectional":
            self._stale_dist = True

    # Standard mode specific methods
    def get_task_ema_stats(self, task_id: int) -> Optional[tuple[float, float, int]]:
        """Get EMA statistics for a task (for debugging/analysis). Standard mode only."""
        if self.mode != "standard":
            return None
        return self._task_emas.get(task_id)

    # Bidirectional mode specific methods
    def get_bidirectional_stats(self) -> Dict[str, Optional[np.ndarray]]:
        """Get bidirectional learning progress statistics (for debugging/analysis). Bidirectional mode only."""
        if self.mode != "bidirectional":
            return {}

        return {
            "p_fast": self._p_fast,
            "p_slow": self._p_slow,
            "p_true": self._p_true,
            "random_baseline": self._random_baseline,
            "task_success_rate": self._task_success_rate,
            "sample_levels": self._sample_levels,
            "task_dist": self._task_dist,
        }

    def get_stats(self) -> Dict[str, float]:
        """Get learning progress statistics (for backward compatibility)."""
        if self.mode == "bidirectional":
            # Provide basic bidirectional stats
            stats = {
                "num_tracked_tasks": float(len(self._outcomes)) if hasattr(self, "_outcomes") else 0.0,
                "mean_task_success_rate": 0.0,
            }

            if hasattr(self, "_task_success_rate") and len(self._task_success_rate) > 0:
                stats["mean_task_success_rate"] = float(self._task_success_rate.mean())

            return stats
        else:
            # Provide basic standard stats
            return {
                "num_tracked_tasks": float(len(self._task_emas)) if hasattr(self, "_task_emas") else 0.0,
                "mean_num_samples": 0.0,
                "mean_ema_score": 0.0,
                "mean_learning_progress": 0.0,
            }

    def _calculate_task_distribution(self) -> None:
        """Calculate task selection distribution based on learning progress. Bidirectional mode only."""
        if self.mode != "bidirectional" or not self._outcomes:
            if self.mode == "bidirectional":
                self._task_dist = np.array([])
                self._stale_dist = False
            return

        task_ids = list(self._outcomes.keys())
        num_tasks = len(task_ids)

        # Initialize arrays if needed
        if self._p_fast is None or len(self._p_fast) != num_tasks:
            self._p_fast = np.full(num_tasks, DEFAULT_SUCCESS_RATE)
            self._p_slow = np.full(num_tasks, DEFAULT_SUCCESS_RATE)
            self._p_true = np.full(num_tasks, DEFAULT_SUCCESS_RATE)
            self._random_baseline = np.full(num_tasks, DEFAULT_SUCCESS_RATE)
            self._task_success_rate = np.full(num_tasks, DEFAULT_SUCCESS_RATE)
            self._update_mask = np.zeros(num_tasks, dtype=bool)
            self._sample_levels = np.zeros(num_tasks)

        # Calculate current success rates and update masks
        for i, task_id in enumerate(task_ids):
            outcomes = self._outcomes[task_id]
            if outcomes:
                self._task_success_rate[i] = np.mean(outcomes)
                self._sample_levels[i] = len(outcomes)
                self._update_mask[i] = len(outcomes) >= self.sample_threshold
            else:
                self._task_success_rate[i] = DEFAULT_SUCCESS_RATE
                self._sample_levels[i] = 0
                self._update_mask[i] = False

        # Update EMAs where we have sufficient samples
        if np.any(self._update_mask):
            # Only update tasks with sufficient samples
            normalized_task_success_rates = self._task_success_rate[self._update_mask]

            # Update fast and slow EMAs
            self._p_fast[self._update_mask] = normalized_task_success_rates * self.ema_timescale + self._p_fast[
                self._update_mask
            ] * (1.0 - self.ema_timescale)

            self._p_slow[self._update_mask] = self._p_fast[self._update_mask] * self.ema_timescale + self._p_slow[
                self._update_mask
            ] * (1.0 - self.ema_timescale)

        # Calculate learning progress as abs difference between fast and slow EMAs
        learning_progress = np.abs(self._p_fast - self._p_slow)

        # Apply sigmoid normalization
        progress_smoothed = 1.0 / (1.0 + np.exp(-learning_progress / self.progress_smoothing))

        # Add exploration bonus for under-sampled tasks
        exploration_bonus_mask = self._sample_levels < self.sample_threshold
        progress_smoothed[exploration_bonus_mask] += self.exploration_bonus

        # Set task distribution
        self._task_dist = progress_smoothed

        self._stale_dist = False


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
        # Initialize parent class which includes bucket_analyzer
        super().__init__(num_tasks, hypers)

        self.num_tasks = num_tasks
        self.hypers = hypers

        # Override task tracker with learning progress specific configuration
        self.task_tracker = TaskTracker(
            max_memory_tasks=hypers.max_memory_tasks,
            max_bucket_axes=hypers.max_bucket_axes,
            logging_detailed_slices=hypers.logging_detailed_slices,
        )

        # Override bucket analyzer with learning progress specific configuration
        self.bucket_analyzer = BucketAnalyzer(
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
        # Call parent method which handles task tracker and bucket analyzer
        super().on_task_evicted(task_id)

        # Remove from learning progress scorer
        self.lp_scorer.remove_task(task_id)

        # Invalidate stats cache when task state changes
        self._stats_cache_valid = False

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance across all components."""
        # Call parent method which handles task tracker and bucket analyzer
        super().update_task_performance(task_id, score)

        # Update learning progress scorer
        self.lp_scorer.update_task_ema(task_id, score)

        # Invalidate stats cache when performance updates
        self._stats_cache_valid = False

    def on_task_created(self, task: CurriculumTask) -> None:
        """Handle new task creation."""
        # Call parent method which handles task tracker
        super().on_task_created(task)

        # Invalidate stats cache when new tasks are created
        self._stats_cache_valid = False

    def stats(self, prefix: str = "") -> Dict[str, float]:
        """Get comprehensive statistics from all components."""
        # Use cached stats if valid to avoid expensive recomputation
        cache_key = f"stats_{prefix}"
        if self._stats_cache_valid and cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        # Get base stats from parent (includes tracker and bucket analyzer)
        stats = super().stats(prefix)

        # Add learning progress scorer specific stats
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
        for k, v in basic_lp_stats.items():
            stats[f"{prefix}lp/{k}"] = v

        # Basic stats for any scorer
        stats[f"{prefix}lp/scorer_mode_id"] = 1.0 if self.lp_scorer.mode == "bidirectional" else 0.0

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
                return np.random.choice(task_ids, p=probabilities)
        else:
            # Use curriculum's RNG for deterministic behavior
            if self._curriculum is not None:
                return self._curriculum._rng.choice(task_ids)
            else:
                # Fallback to random for backwards compatibility
                return random.choice(task_ids)
