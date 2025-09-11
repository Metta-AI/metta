import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .curriculum import CurriculumAlgorithm, CurriculumAlgorithmConfig, CurriculumTask
from .task_tracker import TaskTracker

logger = logging.getLogger(__name__)

# Constants for bidirectional learning progress
DEFAULT_SUCCESS_RATE = 0.0
DEFAULT_WEIGHT = 1.0
RANDOM_BASELINE_CAP = 0.75


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


class _LearningProgressScorer:
    """Unified learning progress scorer that supports both variance and bidirectional modes."""

    def __init__(
        self,
        ema_timescale: float = 0.001,
        exploration_bonus: float = 0.1,
        use_bidirectional: bool = True,
        progress_smoothing: float = 0.05,
        num_active_tasks: int = 16,
        rand_task_rate: float = 0.25,
        sample_threshold: int = 10,
        memory: int = 25,
    ):
        self.ema_timescale = ema_timescale
        self.exploration_bonus = exploration_bonus
        self.use_bidirectional = use_bidirectional
        self.progress_smoothing = progress_smoothing
        self.num_active_tasks = num_active_tasks
        self.rand_task_rate = rand_task_rate
        self.sample_threshold = sample_threshold
        self.memory = memory

        if use_bidirectional:
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
        else:
            # Variance-based learning progress tracking
            # EMA tracking for each task: task_id -> (ema_score, ema_squared, num_samples)
            self._task_emas: Dict[int, tuple[float, float, int]] = {}

        # Common cache for learning progress scores
        self._score_cache: Dict[int, float] = {}
        self._cache_valid_tasks: set[int] = set()

    def update_task_ema(self, task_id: int, score: float) -> None:
        """Update EMA tracking for a task with new score."""
        if self.use_bidirectional:
            self._update_bidirectional_ema(task_id, score)
        else:
            self._update_variance_ema(task_id, score)

    def _update_bidirectional_ema(self, task_id: int, score: float) -> None:
        """Update bidirectional EMA tracking for a task with new score."""
        # Convert score to success rate (assuming score is between 0 and 1)
        success_rate = max(0.0, min(1.0, score))

        # Initialize outcomes for new tasks
        if task_id not in self._outcomes:
            self._outcomes[task_id] = []

        # Add outcome and maintain memory limit
        self._outcomes[task_id].append(success_rate)
        self._outcomes[task_id] = self._outcomes[task_id][-self.memory :]

        # Update counter
        if task_id not in self._counter:
            self._counter[task_id] = 0
        self._counter[task_id] += 1

        # Mark distribution as stale
        self._stale_dist = True
        self._cache_valid_tasks.discard(task_id)

    def _update_variance_ema(self, task_id: int, score: float) -> None:
        """Update variance-based EMA tracking for a task with new score."""
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
        """Calculate learning progress score for a task using either bidirectional or variance method."""
        # Return cached score if valid
        if task_id in self._cache_valid_tasks and task_id in self._score_cache:
            return self._score_cache[task_id]

        if self.use_bidirectional:
            score = self._get_bidirectional_score(task_id, task_tracker)
        else:
            score = self._get_variance_score(task_id, task_tracker)

        # Cache the computed score
        self._score_cache[task_id] = score
        self._cache_valid_tasks.add(task_id)
        return score

    def _get_bidirectional_score(self, task_id: int, task_tracker: TaskTracker) -> float:
        """Calculate bidirectional learning progress score for a task."""
        task_stats = task_tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < 2:
            # New tasks get exploration bonus
            return self.exploration_bonus
        elif task_id not in self._outcomes or len(self._outcomes[task_id]) < 2:
            # Tasks without sufficient data get exploration bonus
            return self.exploration_bonus
        else:
            # Calculate bidirectional learning progress
            self._update_bidirectional_progress()

            # Get task distribution if needed
            if self._task_dist is None or self._stale_dist:
                self._calculate_task_distribution()

            # Find task index in our tracking
            task_indices = list(self._outcomes.keys())
            if task_id in task_indices:
                task_idx = task_indices.index(task_id)
                if self._task_dist is not None and task_idx < len(self._task_dist):
                    # Use the bidirectional learning progress as score
                    return float(self._task_dist[task_idx])
                else:
                    return self.exploration_bonus
            else:
                return self.exploration_bonus

    def _get_variance_score(self, task_id: int, task_tracker: TaskTracker) -> float:
        """Calculate variance-based learning progress score for a task."""
        task_stats = task_tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < 2:
            # New tasks get exploration bonus
            return self.exploration_bonus
        elif task_id not in self._task_emas:
            # Tasks without EMA tracking get exploration bonus (they're new to the scorer)
            return self.exploration_bonus
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

            return learning_progress

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
        if self.use_bidirectional:
            self._outcomes.pop(task_id, None)
            self._counter.pop(task_id, None)
            self._stale_dist = True
        else:
            self._task_emas.pop(task_id, None)

        self._score_cache.pop(task_id, None)
        self._cache_valid_tasks.discard(task_id)

    def get_stats(self) -> Dict[str, float]:
        """Get statistics about learning progress scoring."""
        if self.use_bidirectional:
            return self._get_bidirectional_stats()
        else:
            return self._get_variance_stats()

    def _get_bidirectional_stats(self) -> Dict[str, float]:
        """Get bidirectional learning progress statistics."""
        if not self._outcomes:
            return {
                "num_tracked_tasks": 0,
                "mean_task_success_rate": 0.0,
                "mean_learning_progress": 0.0,
                "num_active_tasks": 0,
            }

        self._update_bidirectional_progress()

        stats = {
            "num_tracked_tasks": len(self._outcomes),
            "mean_task_success_rate": float(np.mean(self._task_success_rate))
            if len(self._task_success_rate) > 0
            else 0.0,
        }

        if self._task_dist is not None and len(self._task_dist) > 0:
            stats.update(
                {
                    "mean_sample_prob": float(np.mean(self._task_dist)),
                    "num_zeros_lp_dist": int(np.sum(self._task_dist == 0)),
                    "mean_learning_progress": float(np.mean(self._learning_progress())),
                }
            )
        else:
            stats.update(
                {
                    "mean_sample_prob": 0.0,
                    "num_zeros_lp_dist": 0,
                    "mean_learning_progress": 0.0,
                }
            )

        return stats

    def _get_variance_stats(self) -> Dict[str, float]:
        """Get variance-based learning progress statistics."""
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
            "num_tracked_tasks": float(len(self._task_emas)),
            "mean_num_samples": float(np.mean(num_samples_list)),
            "mean_ema_score": float(np.mean(ema_scores)),
            "mean_learning_progress": float(np.mean(learning_progress_scores)) if learning_progress_scores else 0.0,
        }

    # Bidirectional-specific helper methods
    def _update_bidirectional_progress(self):
        """Update bidirectional learning progress tracking with current task success rates."""
        if not self.use_bidirectional or not self._outcomes:
            return

        # Get all tracked task IDs
        task_ids = sorted(self._outcomes.keys())
        num_tasks = len(task_ids)

        if num_tasks == 0:
            return

        # Calculate task success rates
        task_success_rates = np.array(
            [
                np.mean(self._outcomes[task_id]) if self._outcomes[task_id] else DEFAULT_SUCCESS_RATE
                for task_id in task_ids
            ]
        )

        # Handle NaN values
        task_success_rates = np.nan_to_num(task_success_rates, nan=DEFAULT_SUCCESS_RATE)

        # Initialize random baseline if needed
        if self._random_baseline is None or len(self._random_baseline) != num_tasks:
            self._random_baseline = np.minimum(task_success_rates, RANDOM_BASELINE_CAP)

        # Create update mask for tasks with sufficient data
        self._update_mask = np.array([len(self._outcomes[task_id]) >= 2 for task_id in task_ids])

        if not np.any(self._update_mask):
            return

        # Handle division by zero in normalization
        denominator = 1.0 - self._random_baseline[self._update_mask]
        denominator = np.where(denominator <= 0, 1.0, denominator)

        normalized_task_success_rates = (
            np.maximum(
                task_success_rates[self._update_mask] - self._random_baseline[self._update_mask],
                np.zeros(task_success_rates[self._update_mask].shape),
            )
            / denominator
        )

        # Initialize or update fast and slow EMAs
        if self._p_fast is None or len(self._p_fast) != num_tasks:
            self._p_fast = np.zeros(num_tasks)
            self._p_slow = np.zeros(num_tasks)
            self._p_true = np.zeros(num_tasks)

            self._p_fast[self._update_mask] = normalized_task_success_rates
            self._p_slow[self._update_mask] = normalized_task_success_rates
            self._p_true[self._update_mask] = task_success_rates[self._update_mask]
        else:
            # Resize arrays if needed
            if self._p_fast is not None and len(self._p_fast) != num_tasks:
                new_p_fast = np.zeros(num_tasks)
                new_p_slow = np.zeros(num_tasks)
                new_p_true = np.zeros(num_tasks)

                min_len = min(len(self._p_fast), num_tasks)
                if self._p_slow is not None and self._p_true is not None:
                    new_p_fast[:min_len] = self._p_fast[:min_len]
                    new_p_slow[:min_len] = self._p_slow[:min_len]
                    new_p_true[:min_len] = self._p_true[:min_len]

                self._p_fast = new_p_fast
                self._p_slow = new_p_slow
                self._p_true = new_p_true

            # Update EMAs (use safe array access)
            if self._p_fast is not None and self._p_slow is not None and self._p_true is not None:
                self._p_fast[self._update_mask] = normalized_task_success_rates * self.ema_timescale + self._p_fast[
                    self._update_mask
                ] * (1.0 - self.ema_timescale)
                self._p_slow[self._update_mask] = self._p_fast[self._update_mask] * self.ema_timescale + self._p_slow[
                    self._update_mask
                ] * (1.0 - self.ema_timescale)
                self._p_true[self._update_mask] = task_success_rates[
                    self._update_mask
                ] * self.ema_timescale + self._p_true[self._update_mask] * (1.0 - self.ema_timescale)

        self._task_success_rate = task_success_rates
        self._stale_dist = True

    def _learning_progress(self, reweight: bool = True) -> np.ndarray:
        """Calculate learning progress as the difference between fast and slow moving averages."""
        if not self.use_bidirectional or self._p_fast is None or self._p_slow is None:
            return np.array([])

        fast = self._reweight(self._p_fast) if reweight else self._p_fast
        slow = self._reweight(self._p_slow) if reweight else self._p_slow
        return np.abs(fast - slow)

    def _reweight(self, probs: np.ndarray) -> np.ndarray:
        """Apply progress smoothing reweighting to probability values."""
        numerator = probs * (1.0 - self.progress_smoothing)
        denominator = probs + self.progress_smoothing * (1.0 - 2.0 * probs)

        # Handle division by zero
        denominator = np.where(denominator <= 0, 1.0, denominator)
        result = numerator / denominator
        return result

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid function to array values."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

    def _calculate_task_distribution(self):
        """Calculate task distribution based on bidirectional learning progress."""
        if not self.use_bidirectional or not self._outcomes:
            self._task_dist = np.array([])
            self._stale_dist = False
            return

        num_tasks = len(self._outcomes)
        task_dist = np.ones(num_tasks) / num_tasks

        learning_progress = self._learning_progress()

        if len(learning_progress) == 0:
            self._task_dist = task_dist
            self._stale_dist = False
            return

        # Find tasks with positive learning progress or true performance
        posidxs = [
            i
            for i, lp in enumerate(learning_progress)
            if lp > 0 or (self._p_true is not None and i < len(self._p_true) and self._p_true[i] > 0)
        ]

        any_progress = len(posidxs) > 0
        subprobs = learning_progress[posidxs] if any_progress else learning_progress

        # Standardize and apply sigmoid
        std = np.std(subprobs)
        if std > 0:
            subprobs = (subprobs - np.mean(subprobs)) / std
        else:
            subprobs = subprobs - np.mean(subprobs)

        subprobs = self._sigmoid(subprobs)

        # Normalize to sum to 1
        sum_probs = np.sum(subprobs)
        if sum_probs > 0:
            subprobs = subprobs / sum_probs
        else:
            subprobs = np.ones_like(subprobs) / len(subprobs)

        # Assign probabilities
        if any_progress:
            task_dist = np.zeros(len(learning_progress))
            task_dist[posidxs] = subprobs
        else:
            task_dist = subprobs

        self._task_dist = task_dist.astype(np.float32)
        self._stale_dist = False


class LearningProgressAlgorithm(CurriculumAlgorithm):
    """
    Learning Progress Algorithm with integrated scoring functionality.

    Combines task tracking, learning progress scoring, and bucket analysis
    into a cohesive algorithm for intelligent task selection based on
    performance variance and exploration needs.
    """

    def __init__(self, num_tasks: int, hypers: LearningProgressConfig):
        super().__init__(num_tasks, hypers, initialize_weights=False)

        self.hypers = hypers

        # Override the default task tracker with learning progress and bucket analysis parameters
        self.task_tracker = TaskTracker(
            max_memory_tasks=hypers.max_memory_tasks,
            max_bucket_axes=hypers.max_bucket_axes,
            logging_detailed_slices=hypers.logging_detailed_slices,
        )

        # Create unified learning progress scorer with all parameters
        self.lp_scorer = _LearningProgressScorer(
            ema_timescale=hypers.ema_timescale,
            exploration_bonus=hypers.exploration_bonus,
            use_bidirectional=hypers.use_bidirectional,
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

    def get_learning_progress_score(self, task_id: int, task_tracker=None) -> float:
        """Public interface for getting learning progress score for a task.

        Args:
            task_id: The task ID to score
            task_tracker: Optional task tracker (defaults to self.task_tracker)

        Returns:
            Learning progress score for the task
        """
        tracker = task_tracker or self.task_tracker
        return self.lp_scorer.get_learning_progress_score(task_id, tracker)

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
        # Remove from task tracker
        self.task_tracker.remove_task(task_id)

        # Learning progress specific cleanup
        self.lp_scorer.remove_task(task_id)

        # Invalidate stats cache when task state changes
        self._stats_cache_valid = False

    def update_task_performance(self, task_id: int, score: float, bucket_values: Dict[str, Any] = None) -> None:
        """Update task performance across all components."""
        # Call parent implementation but pass bucket_values to task_tracker directly
        if task_id not in self.task_tracker._task_memory:
            self.task_tracker.track_task_creation(task_id)

        # Update task tracker with bucket values
        self.task_tracker.update_task_performance(task_id, score, bucket_values or {})

        # Update learning progress EMA
        self.lp_scorer.update_task_ema(task_id, score)

        # Invalidate stats cache when performance updates
        self._stats_cache_valid = False

    def on_task_created(self, task: CurriculumTask) -> None:
        """Handle new task creation."""
        task_id = task._task_id

        # Track task creation directly
        self.task_tracker.track_task_creation(task_id)

        # Extract and track bucket values using integrated TaskTracker
        bucket_values = self.task_tracker.extract_bucket_values(task)
        if bucket_values:
            # Initialize bucket tracking with default score
            self.task_tracker.update_task_performance(task_id, 0.0, bucket_values)

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

        # Bucket analysis stats (now integrated into task tracker)
        # The bucket stats are already included in the parent stats() call via TaskTracker.get_global_stats()

        # Detailed bucket density stats (if enabled) - this is expensive
        if hasattr(self.hypers, "logging_detailed_slices") and self.hypers.logging_detailed_slices:
            density_stats = self.task_tracker.get_completion_density_stats()
            for bucket_name, bucket_stats in density_stats.items():
                bucket_prefix = f"bucket_{bucket_name}/"
                stats.update(add_prefix(bucket_stats, bucket_prefix))

        # Cache the result
        self._stats_cache[prefix] = stats
        self._stats_cache_valid = True

        return stats

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
