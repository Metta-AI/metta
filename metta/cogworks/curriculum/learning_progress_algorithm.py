"""
Learning Progress Algorithm with integrated bidirectional scoring.

Provides intelligent task selection based on bidirectional learning progress analysis,
using fast and slow exponential moving averages to detect learning opportunities.
"""

import random
from typing import Any, Dict, List, Optional

import numpy as np

from .curriculum import CurriculumAlgorithm, CurriculumAlgorithmConfig, CurriculumTask
from .task_tracker import CentralizedTaskTracker, LocalTaskTracker, TaskTracker

# Constants for bidirectional learning progress
DEFAULT_SUCCESS_RATE = 0.0
DEFAULT_WEIGHT = 1.0
RANDOM_BASELINE_CAP = 0.75


class LearningProgressConfig(CurriculumAlgorithmConfig):
    """Configuration for learning progress with bidirectional scoring as default."""

    type: str = "learning_progress"

    # Bidirectional learning progress settings (now default)
    use_bidirectional: bool = True
    ema_timescale: float = 0.001
    slow_timescale_factor: float = 0.2  # Multiplier for slow EMA timescale (slow = ema_timescale * this)
    exploration_bonus: float = 0.1
    progress_smoothing: float = 0.05  # For bidirectional reweighting
    performance_bonus_weight: float = 0.0  # Weight for performance bonus in LP calculation

    # Task distribution and sampling
    num_active_tasks: int = 16
    rand_task_rate: float = 0.25
    sample_threshold: int = 10
    memory: int = 25
    eviction_threshold_percentile: float = 0.4  # Bottom percentile for task eviction

    # Basic EMA mode parameters (when use_bidirectional=False)
    basic_ema_initial_alpha: float = 0.3  # Initial learning rate for basic EMA
    basic_ema_alpha_decay: float = 0.2  # Decay factor for basic EMA alpha
    exploration_blend_factor: float = 0.5  # Blend factor for exploration in basic mode

    # Task tracker EMA configuration
    task_tracker_ema_alpha: float = 0.1  # Learning rate for task tracker EMAs (reward, success rate)

    # Performance and memory management
    max_memory_tasks: int = 1000
    max_slice_axes: int = 3  # Updated terminology

    # Memory backend configuration
    task_struct_size: int = 12  # Size of task data structure in shared memory
    completion_history_size: int = 1000  # Size of completion history array
    enable_detailed_slice_logging: bool = False  # Updated terminology
    use_shared_memory: bool = True  # Enabled by default for production use
    session_id: Optional[str] = None  # Session ID for shared memory, None = auto-generate unique

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

        # Initialize task tracker with appropriate implementation
        if hypers.use_shared_memory:
            self.task_tracker: TaskTracker = CentralizedTaskTracker(
                max_memory_tasks=hypers.max_memory_tasks,
                session_id=hypers.session_id,
                ema_alpha=hypers.task_tracker_ema_alpha,
                task_struct_size=hypers.task_struct_size,
                completion_history_size=hypers.completion_history_size,
            )
        else:
            self.task_tracker = LocalTaskTracker(
                max_memory_tasks=hypers.max_memory_tasks,
                ema_alpha=hypers.task_tracker_ema_alpha,
            )

        # Note: slice_analyzer is already initialized in parent class via StatsLogger

        # Initialize scoring method (bidirectional by default)
        if hypers.use_bidirectional:
            self._init_bidirectional_scoring()
        else:
            self._init_basic_scoring()

        # Cache for expensive statistics computation
        self._stats_cache: Dict[str, Any] = {}
        self._stats_cache_valid = False

    @property
    def lp_scorer(self):
        """Compatibility property for tests that expect lp_scorer attribute."""
        return self

    @property
    def exploration_bonus(self):
        """Compatibility property for tests that expect exploration_bonus attribute."""
        return self.hypers.exploration_bonus

    def get_base_stats(self) -> Dict[str, float]:
        """Get basic statistics that all algorithms must provide."""
        base_stats = {"num_tasks": self.num_tasks, **self.slice_analyzer.get_base_stats()}

        # Add task tracker stats with prefix for test compatibility
        tracker_stats = self.task_tracker.get_global_stats()
        for key, value in tracker_stats.items():
            base_stats[f"tracker/{key}"] = value

        return base_stats

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

    def _init_bidirectional_scoring(self):
        """Initialize bidirectional EMA tracking (integrated from BidirectionalLearningProgressScorer)."""
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
        self._score_cache: Dict[int, float] = {}
        self._cache_valid_tasks: set[int] = set()

    def _init_basic_scoring(self):
        """Initialize basic EMA tracking (fallback method)."""
        # EMA tracking for each task: task_id -> (ema_score, ema_squared, num_samples)
        self._task_emas: Dict[int, tuple[float, float, int]] = {}
        # Initialize cache for basic scoring mode
        self._score_cache = getattr(self, "_score_cache", {})
        self._cache_valid_tasks = getattr(self, "_cache_valid_tasks", set())

    def score_tasks(self, task_ids: List[int]) -> Dict[int, float]:
        """Score tasks using the configured method (bidirectional by default)."""
        if self.hypers.use_bidirectional:
            return self._score_tasks_bidirectional(task_ids)
        else:
            return self._score_tasks_basic(task_ids)

    def _score_tasks_bidirectional(self, task_ids: List[int]) -> Dict[int, float]:
        """Score tasks using bidirectional learning progress."""
        scores = {}
        for task_id in task_ids:
            scores[task_id] = self._get_bidirectional_learning_progress_score(task_id)
        return scores

    def _score_tasks_basic(self, task_ids: List[int]) -> Dict[int, float]:
        """Score tasks using basic EMA variance method."""
        scores = {}
        for task_id in task_ids:
            scores[task_id] = self._get_basic_learning_progress_score(task_id)
        return scores

    def _get_bidirectional_learning_progress_score(self, task_id: int) -> float:
        """Calculate bidirectional learning progress score for a task."""
        # Return cached score if valid
        if task_id in self._cache_valid_tasks and task_id in self._score_cache:
            return self._score_cache[task_id]

        task_stats = self.task_tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < 2:
            # New tasks get exploration bonus
            score = self.hypers.exploration_bonus
        elif task_id not in self._outcomes or len(self._outcomes[task_id]) < 2:
            # Tasks without sufficient data get exploration bonus
            score = self.hypers.exploration_bonus
        else:
            # Calculate bidirectional learning progress
            self._update_bidirectional_progress()

            # Get task distribution if needed
            if self._task_dist is None or self._stale_dist:
                self._calculate_task_distribution()

            # Find task index in our tracking
            task_indices = list(self._outcomes.keys())
            if task_id in task_indices and self._task_dist is not None:
                task_idx = task_indices.index(task_id)
                if task_idx < len(self._task_dist):
                    # Use the bidirectional learning progress as score
                    score = float(self._task_dist[task_idx])
                else:
                    score = self.hypers.exploration_bonus
            else:
                score = self.hypers.exploration_bonus

        # Cache the computed score
        self._score_cache[task_id] = score
        self._cache_valid_tasks.add(task_id)
        return score

    def _get_basic_learning_progress_score(self, task_id: int) -> float:
        """Calculate basic learning progress score using EMA variance."""
        # Return cached score if valid
        if task_id in self._cache_valid_tasks and task_id in self._score_cache:
            return self._score_cache[task_id]

        task_stats = self.task_tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < 2:
            score = self.hypers.exploration_bonus
        elif task_id not in self._task_emas:
            score = self.hypers.exploration_bonus
        else:
            ema_score, ema_squared, num_samples = self._task_emas[task_id]

            # Calculate variance from EMA
            variance = max(0.0, ema_squared - ema_score * ema_score)
            std_dev = np.sqrt(variance)

            # Learning progress is approximated by variance in performance
            learning_progress = std_dev

            # For tasks with few samples, blend variance with exploration bonus
            if num_samples < 10:
                exploration_weight = (10 - num_samples) / 10
                exploration_bonus = self.hypers.exploration_bonus * exploration_weight
                # Blend variance-based score with exploration bonus, favoring variance as we get more data
                learning_progress = (
                    learning_progress * (1 - exploration_weight * self.hypers.exploration_blend_factor)
                    + exploration_bonus
                )

            score = learning_progress

        # Cache the computed score
        self._score_cache[task_id] = score
        self._cache_valid_tasks.add(task_id)
        return score

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

        # Invalidate stats cache when task state changes
        self.invalidate_cache()

    def _remove_task_from_scoring(self, task_id: int) -> None:
        """Remove task from scoring system."""
        if self.hypers.use_bidirectional:
            self._outcomes.pop(task_id, None)
            self._counter.pop(task_id, None)
            self._score_cache.pop(task_id, None)
            self._cache_valid_tasks.discard(task_id)
            self._stale_dist = True
        else:
            self._task_emas.pop(task_id, None)
            self._score_cache.pop(task_id, None)
            self._cache_valid_tasks.discard(task_id)

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance using the appropriate scoring method."""
        # Update EMA tracking first
        if self.hypers.use_bidirectional:
            self._update_bidirectional_ema(task_id, score)
        else:
            self._update_basic_ema(task_id, score)

        # Clear cache to ensure fresh calculation
        self._cache_valid_tasks.discard(task_id)
        self._score_cache.pop(task_id, None)

        # Calculate LP score based on updated EMAs
        if self.hypers.use_bidirectional:
            lp_score = self._get_bidirectional_learning_progress_score(task_id)
        else:
            lp_score = self._get_basic_learning_progress_score(task_id)

        # Single atomic update to task tracker with both score and LP score
        # This ensures consistency and avoids multiple writes to shared memory
        self.task_tracker.update_task_performance(task_id, score, lp_score=lp_score)

        # Invalidate stats cache
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

    def get_learning_progress_score(self, task_id: int, task_tracker=None) -> float:
        """Get learning progress score for a specific task (compatibility method for tests)."""
        if self.hypers.use_bidirectional:
            return self._get_bidirectional_learning_progress_score(task_id)
        else:
            return self._get_basic_learning_progress_score(task_id)

    def get_stats(self) -> Dict[str, float]:
        """Get learning progress statistics (compatibility method for tests)."""
        if self.hypers.use_bidirectional:
            return self._get_bidirectional_detailed_stats()
        else:
            return self._get_basic_detailed_stats()

    def update_task_with_slice_values(self, task_id: int, score: float, slice_values: Dict[str, Any]) -> None:
        """Update task performance including slice values for analysis."""
        # First update performance
        self.update_task_performance(task_id, score)

        # Then update slice analyzer
        self.slice_analyzer.update_task_completion(task_id, slice_values, score)

    def _update_bidirectional_ema(self, task_id: int, score: float) -> None:
        """Update bidirectional EMA tracking for a task with new score."""
        # Convert score to success rate (assuming score is between 0 and 1)
        success_rate = max(0.0, min(1.0, score))

        # Initialize outcomes for new tasks
        if task_id not in self._outcomes:
            self._outcomes[task_id] = []

        # Add outcome and maintain memory limit
        self._outcomes[task_id].append(success_rate)
        self._outcomes[task_id] = self._outcomes[task_id][-self.hypers.memory :]

        # Update counter
        if task_id not in self._counter:
            self._counter[task_id] = 0
        self._counter[task_id] += 1

        # Update bidirectional progress to ensure EMAs are updated
        self._update_bidirectional_progress()

        # Mark distribution as stale
        self._stale_dist = True
        self._cache_valid_tasks.discard(task_id)

    def _update_basic_ema(self, task_id: int, score: float) -> None:
        """Update basic EMA tracking for a task with new score."""
        if task_id not in self._task_emas:
            self._task_emas[task_id] = (score, score * score, 1)
        else:
            ema_score, ema_squared, num_samples = self._task_emas[task_id]

            # Update EMAs with more responsive learning rate for small sample sizes
            # Use higher learning rate initially, then decay to configured timescale
            if num_samples < 10:
                # Start with higher learning rate for better sensitivity
                base_alpha = self.hypers.basic_ema_initial_alpha / (1 + num_samples * self.hypers.basic_ema_alpha_decay)
            else:
                # Use configured timescale for larger samples
                base_alpha = self.hypers.ema_timescale * num_samples

            alpha = min(1.0, base_alpha)
            new_ema_score = (1 - alpha) * ema_score + alpha * score
            new_ema_squared = (1 - alpha) * ema_squared + alpha * (score * score)

            self._task_emas[task_id] = (new_ema_score, new_ema_squared, num_samples + 1)

        # Invalidate cache for this task when EMA is updated
        self._cache_valid_tasks.discard(task_id)

    def on_task_created(self, task: CurriculumTask) -> None:
        """Handle task creation by tracking it."""
        self.task_tracker.track_task_creation(task._task_id)

        # Extract and update slice values if available
        slice_values = task.get_slice_values()
        if slice_values:
            # Initial tracking with neutral score
            self.slice_analyzer.update_task_completion(task._task_id, slice_values, 0.5)

        # Invalidate stats cache when task state changes
        self.invalidate_cache()

    def get_detailed_stats(self) -> Dict[str, float]:
        """Get detailed stats including learning progress and slice distribution analysis."""
        stats = super().get_detailed_stats()  # Gets slice analyzer stats

        # Always include learning progress stats (not just when detailed logging is enabled)
        if self.hypers.use_bidirectional:
            lp_stats = self._get_bidirectional_detailed_stats()
        else:
            lp_stats = self._get_basic_detailed_stats()

        # Add lp/ prefix to learning progress stats
        for key, value in lp_stats.items():
            stats[f"lp/{key}"] = value

        return stats

    def _get_bidirectional_detailed_stats(self) -> Dict[str, float]:
        """Get detailed bidirectional learning progress statistics."""
        if not self._outcomes:
            return {
                "mean_task_success_rate": 0.0,
                "mean_learning_progress": 0.0,
            }

        self._update_bidirectional_progress()

        stats = {
            "mean_task_success_rate": float(np.mean(self._task_success_rate))
            if len(self._task_success_rate) > 0
            else 0.0,
        }

        if self._task_dist is not None and len(self._task_dist) > 0:
            stats.update(
                {
                    "mean_sample_prob": float(np.mean(self._task_dist)),
                    "num_zeros_lp_dist": float(np.sum(self._task_dist == 0)),
                    "mean_learning_progress": float(np.mean(self._learning_progress())),
                }
            )
        else:
            stats.update(
                {
                    "mean_sample_prob": 0.0,
                    "num_zeros_lp_dist": 0.0,
                    "mean_learning_progress": 0.0,
                }
            )

        return stats

    def _get_basic_detailed_stats(self) -> Dict[str, float]:
        """Get detailed basic learning progress statistics."""
        if not self._task_emas:
            return {
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
            "mean_num_samples": float(np.mean(num_samples_list)),
            "mean_ema_score": float(np.mean(ema_scores)),
            "mean_learning_progress": float(np.mean(learning_progress_scores)) if learning_progress_scores else 0.0,
        }

    # Bidirectional learning progress implementation (integrated from modules)

    def _update_bidirectional_progress(self):
        """Update bidirectional learning progress tracking with current task success rates."""
        if not self._outcomes:
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
            # Random baseline should represent baseline/random performance, typically around 0.5
            # Don't use actual task performance as baseline - that defeats the purpose
            self._random_baseline = np.full(num_tasks, 0.5)

        # Create update mask for tasks with sufficient data
        self._update_mask = np.array([len(self._outcomes[task_id]) >= 2 for task_id in task_ids])

        if not np.any(self._update_mask):
            return

        # Handle division by zero in normalization
        denominator = 1.0 - self._random_baseline[self._update_mask]
        denominator = np.where(denominator <= 0, 1.0, denominator)

        # Allow negative normalized rates for bidirectional algorithm
        # This captures the full range of performance relative to baseline
        normalized_task_success_rates = (
            task_success_rates[self._update_mask] - self._random_baseline[self._update_mask]
        ) / denominator

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
            if (
                self._p_fast is not None
                and self._p_slow is not None
                and self._p_true is not None
                and len(self._p_fast) != num_tasks
            ):
                new_p_fast = np.zeros(num_tasks)
                new_p_slow = np.zeros(num_tasks)
                new_p_true = np.zeros(num_tasks)

                min_len = min(len(self._p_fast), num_tasks)
                new_p_fast[:min_len] = self._p_fast[:min_len]
                new_p_slow[:min_len] = self._p_slow[:min_len]
                new_p_true[:min_len] = self._p_true[:min_len]

                self._p_fast = new_p_fast
                self._p_slow = new_p_slow
                self._p_true = new_p_true

            # Update EMAs (with None checks)
            if self._p_fast is not None and self._p_slow is not None and self._p_true is not None:
                # Fast EMA uses the configured timescale
                self._p_fast[self._update_mask] = (
                    normalized_task_success_rates * self.hypers.ema_timescale
                    + self._p_fast[self._update_mask] * (1.0 - self.hypers.ema_timescale)
                )
                # Slow EMA uses a much slower timescale for better differentiation
                slow_timescale = self.hypers.ema_timescale * self.hypers.slow_timescale_factor
                self._p_slow[self._update_mask] = normalized_task_success_rates * slow_timescale + self._p_slow[
                    self._update_mask
                ] * (1.0 - slow_timescale)
                self._p_true[self._update_mask] = task_success_rates[
                    self._update_mask
                ] * self.hypers.ema_timescale + self._p_true[self._update_mask] * (1.0 - self.hypers.ema_timescale)

        self._task_success_rate = task_success_rates
        self._stale_dist = True

    def _learning_progress(self, reweight: bool = True) -> np.ndarray:
        """Calculate learning progress as the difference between fast and slow moving averages."""
        if self._p_fast is None or self._p_slow is None:
            return np.array([])

        fast = self._reweight(self._p_fast) if reweight else self._p_fast
        slow = self._reweight(self._p_slow) if reweight else self._p_slow

        # Learning progress is the absolute difference between fast and slow EMAs
        # This captures variability/change regardless of absolute performance level
        lp = np.abs(fast - slow)

        # Add a small amount based on fast EMA to slightly favor above-baseline tasks
        # but still prioritize change/variance
        performance_bonus = np.maximum(fast, 0) * self.hypers.performance_bonus_weight

        return lp + performance_bonus

    def _reweight(self, probs: np.ndarray) -> np.ndarray:
        """Apply progress smoothing reweighting to probability values."""
        numerator = probs * (1.0 - self.hypers.progress_smoothing)
        denominator = probs + self.hypers.progress_smoothing * (1.0 - 2.0 * probs)

        # Handle division by zero
        denominator = np.where(denominator <= 0, 1.0, denominator)
        result = numerator / denominator
        return result

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid function to array values."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

    def _calculate_task_distribution(self):
        """Calculate task distribution based on bidirectional learning progress."""
        if not self._outcomes:
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

        # Find tasks with positive learning progress (actual learning/change)
        # Don't just reward any positive performance - focus on learning progress
        posidxs = [i for i, lp in enumerate(learning_progress) if lp > 0]

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

    def get_state(self) -> Dict[str, Any]:
        """Get learning progress algorithm state for checkpointing."""
        state = {
            "type": self.hypers.algorithm_type(),
            "hypers": self.hypers.model_dump(),
            "task_tracker": self.task_tracker.get_state(),
        }

        # Save bidirectional scoring state
        if hasattr(self, "_outcomes"):
            state.update(
                {
                    "outcomes": {k: v for k, v in self._outcomes.items()},
                    "counter": self._counter,
                    "p_fast": self._p_fast.tolist() if self._p_fast is not None else None,
                    "p_slow": self._p_slow.tolist() if self._p_slow is not None else None,
                    "p_true": self._p_true.tolist() if self._p_true is not None else None,
                    "random_baseline": self._random_baseline.tolist() if self._random_baseline is not None else None,
                    "task_success_rate": self._task_success_rate.tolist(),
                    "update_mask": self._update_mask.tolist(),
                    "sample_levels": self._sample_levels.tolist(),
                    "task_dist": self._task_dist.tolist() if self._task_dist is not None else None,
                    "stale_dist": self._stale_dist,
                    "score_cache": self._score_cache,
                    "cache_valid_tasks": list(self._cache_valid_tasks),
                }
            )

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load learning progress algorithm state from checkpoint."""
        # Restore task tracker
        self.task_tracker.load_state(state["task_tracker"])

        # Restore bidirectional scoring state
        if "outcomes" in state:
            self._outcomes = state["outcomes"]
            self._counter = state["counter"]
            self._p_fast = np.array(state["p_fast"]) if state["p_fast"] is not None else None
            self._p_slow = np.array(state["p_slow"]) if state["p_slow"] is not None else None
            self._p_true = np.array(state["p_true"]) if state["p_true"] is not None else None
            self._random_baseline = np.array(state["random_baseline"]) if state["random_baseline"] is not None else None
            self._task_success_rate = np.array(state["task_success_rate"])
            self._update_mask = np.array(state["update_mask"])
            self._sample_levels = np.array(state["sample_levels"])
            self._task_dist = np.array(state["task_dist"]) if state["task_dist"] is not None else None
            self._stale_dist = state["stale_dist"]
            self._score_cache = state["score_cache"]
            self._cache_valid_tasks = set(state["cache_valid_tasks"])

    def cleanup_shared_memory(self) -> None:
        """Clean up shared memory resources with better error handling."""
        if not hasattr(self, "task_tracker"):
            return

        try:
            # CentralizedTaskTracker has cleanup_shared_memory method
            if isinstance(self.task_tracker, CentralizedTaskTracker):
                self.task_tracker.cleanup_shared_memory()
        except Exception as e:
            # Log but don't raise - cleanup should be best-effort
            import logging

            logging.warning(f"Failed to cleanup shared memory: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, "hypers") and getattr(self.hypers, "use_shared_memory", False):
            self.cleanup_shared_memory()
