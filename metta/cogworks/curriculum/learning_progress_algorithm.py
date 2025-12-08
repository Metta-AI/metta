"""
Learning Progress Algorithm with integrated bidirectional scoring.

Provides intelligent task selection based on bidirectional learning progress analysis,
using fast and slow exponential moving averages to detect learning opportunities.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .curriculum import CurriculumAlgorithm, CurriculumAlgorithmConfig, CurriculumTask
from .task_tracker import TaskTracker

# Constants for bidirectional learning progress
DEFAULT_SUCCESS_RATE = 0.0
DEFAULT_WEIGHT = 1.0
RANDOM_BASELINE_CAP = 0.75
BASELINE = 0.5
MIN_DENOM = 0.01


class LearningProgressConfig(CurriculumAlgorithmConfig):
    type: str = "learning_progress"

    # Bidirectional learning progress settings (now default)
    use_bidirectional: bool = True
    ema_timescale: float = 0.001
    exploration_bonus: float = 0.1
    progress_smoothing: float = 0.05  # For bidirectional reweighting

    # Task distribution and sampling
    num_active_tasks: int = 1000
    rand_task_rate: float = 0.25
    sample_threshold: int = 10
    memory: int = 25

    # Performance and memory management
    max_memory_tasks: int = 1000
    max_slice_axes: int = 3  # Updated terminology
    enable_detailed_slice_logging: bool = False  # Updated terminology

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

        self.task_tracker = TaskTracker(max_memory_tasks=hypers.max_memory_tasks)

        self._score_cache: Dict[int, float] = {}
        self._cache_valid_tasks: set[int] = set()

        # Cache for expensive statistics computation
        self._stats_cache: Dict[str, Any] = {}
        self._stats_cache_valid = False

        if hypers.use_bidirectional:
            self._init_bidirectional_scoring()
        else:
            self._init_basic_scoring()

    def _normalized_success(self, score: float) -> float:
        clamped = max(0.0, min(1.0, score))
        return (clamped - BASELINE) / max(1.0 - BASELINE, MIN_DENOM)

    def _invalidate_task_cache(self, task_id: int) -> None:
        self._cache_valid_tasks.discard(task_id)
        self._score_cache.pop(task_id, None)

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

        # Always include learning progress stats (not just when detailed logging is enabled)
        if self.hypers.use_bidirectional:
            lp_stats = self._get_bidirectional_detailed_stats()
        else:
            lp_stats = self._get_basic_detailed_stats()

        # Add lp/ prefix to learning progress stats
        for key, value in lp_stats.items():
            stats[f"lp/{key}"] = value

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
        self._counter: Dict[int, int] = {}

        # Per-task EMA dictionaries (replaces parallel arrays)
        self._per_task_fast: Dict[int, float] = {}
        self._per_task_slow: Dict[int, float] = {}

        # Legacy arrays kept for compatibility with distribution calculation
        self._p_fast: Optional[np.ndarray] = None
        self._p_slow: Optional[np.ndarray] = None
        self._p_true: Optional[np.ndarray] = None
        self._random_baseline: Optional[np.ndarray] = None
        self._task_success_rate: np.ndarray = np.array([])
        self._update_mask: np.ndarray = np.array([])
        self._sample_levels: np.ndarray = np.array([])

        # Cache for task distribution and scores
        self._task_dist: Optional[np.ndarray] = None
        self._stale_dist = True

    def _init_basic_scoring(self):
        """Initialize basic EMA tracking (fallback method)."""
        # EMA tracking for each task: task_id -> (ema_score, ema_squared, num_samples)
        self._task_emas: Dict[int, tuple[float, float, int]] = {}

    def score_tasks(self, task_ids: List[int]) -> Dict[int, float]:
        """Score tasks using the configured method (bidirectional by default)."""
        if self.hypers.use_bidirectional:
            return self._score_tasks_bidirectional(task_ids)
        else:
            return self._score_tasks_basic(task_ids)

    def _score_tasks_bidirectional(self, task_ids: List[int]) -> Dict[int, float]:
        """Score tasks with per-task LP, then sigmoid + normalize per call."""
        if not task_ids:
            return {}

        raw_scores = np.array([self._get_bidirectional_learning_progress_score(tid) for tid in task_ids], dtype=float)
        norm_scores = self._normalize_bidirectional_scores(raw_scores)
        return {tid: float(score) for tid, score in zip(task_ids, norm_scores, strict=True)}

    def _score_tasks_basic(self, task_ids: List[int]) -> Dict[int, float]:
        return {task_id: self._get_basic_learning_progress_score(task_id) for task_id in task_ids}

    def _get_bidirectional_learning_progress_score(self, task_id: int) -> float:
        """Calculate bidirectional learning progress score for a task."""
        if task_id in self._cache_valid_tasks and task_id in self._score_cache:
            return self._score_cache[task_id]

        if task_id not in self._per_task_fast or task_id not in self._outcomes or len(self._outcomes[task_id]) < 2:
            score = self.hypers.exploration_bonus
        else:
            lp = abs(self._per_task_fast[task_id] - self._per_task_slow[task_id])
            perf_bonus = max(self._per_task_fast[task_id], 0) * 0.1
            score = max(lp + perf_bonus, self.hypers.exploration_bonus)

        self._score_cache[task_id] = score
        self._cache_valid_tasks.add(task_id)
        return score

    def _get_basic_learning_progress_score(self, task_id: int) -> float:
        """Calculate basic learning progress score using EMA variance."""
        if task_id in self._cache_valid_tasks and task_id in self._score_cache:
            return self._score_cache[task_id]

        task_stats = self.task_tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < 2 or task_id not in self._task_emas:
            score = self.hypers.exploration_bonus
        else:
            ema_score, ema_squared, num_samples = self._task_emas[task_id]
            variance = max(0.0, ema_squared - ema_score * ema_score)
            learning_progress = np.sqrt(variance)
            if num_samples < 10:
                learning_progress += self.hypers.exploration_bonus * (10 - num_samples) / 10
            score = learning_progress

        self._score_cache[task_id] = score
        self._cache_valid_tasks.add(task_id)
        return score

    def recommend_eviction(self, task_ids: List[int]) -> Optional[int]:
        """Recommend which task to evict based on learning progress."""
        if not task_ids:
            return None

        # Use the same scoring signal as sampling; lower score = lower learning progress
        scores = self.score_tasks(task_ids)

        def _evict_key(tid: int) -> tuple[float, int, int]:
            task_stats = self.task_tracker.get_task_stats(tid) or {"completion_count": 0}
            # Fewest presentations is preferred when scores tie; final tie-breaker on task_id for determinism
            return (scores.get(tid, self.hypers.exploration_bonus), task_stats["completion_count"], tid)

        return min(task_ids, key=_evict_key)

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

        # Evict if this task is in the bottom 40% of learning progress scores
        # This ensures eviction happens more readily with small task pools
        sorted_scores = sorted(scores.values())
        threshold_index = max(0, int(len(sorted_scores) * 0.4))
        threshold_score = sorted_scores[threshold_index] if sorted_scores else 0.0

        return task_score <= threshold_score

    def on_task_evicted(self, task_id: int) -> None:
        """Clean up when a task is evicted."""
        # Remove from task tracker
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
            self._per_task_fast.pop(task_id, None)
            self._per_task_slow.pop(task_id, None)
        else:
            self._task_emas.pop(task_id, None)
        self._invalidate_task_cache(task_id)

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance using the appropriate scoring method."""
        # Update task tracker
        self.task_tracker.update_task_performance(task_id, score)

        # Update scoring method
        if self.hypers.use_bidirectional:
            self._update_bidirectional_ema(task_id, score)
        else:
            self._update_basic_ema(task_id, score)

        # Invalidate stats cache
        self.invalidate_cache()

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
        """Update bidirectional EMA tracking for a single task."""
        # Normalize score to [0, 1] range for learning progress calculation
        # Note: This assumes rewards are normalized. If rewards are outside [0,1],
        # consider using a different normalization strategy (e.g., sigmoid or min-max scaling)
        success_rate = max(0.0, min(1.0, score))

        # Initialize outcomes for new tasks
        if task_id not in self._outcomes:
            self._outcomes[task_id] = []

        self._outcomes[task_id].append(success_rate)
        self._outcomes[task_id] = self._outcomes[task_id][-self.hypers.memory :]

        if task_id not in self._counter:
            self._counter[task_id] = 0
        self._counter[task_id] += 1

        # === FIX: Update only THIS task's EMAs ===
        baseline = 0.5  # Fixed baseline, not task-dependent
        denominator = max(1.0 - baseline, 0.01)
        normalized = (success_rate - baseline) / denominator

        # Initialize per-task EMAs if needed
        if task_id not in self._per_task_fast:
            self._per_task_fast[task_id] = normalized
            self._per_task_slow[task_id] = normalized
        else:
            # Fast EMA
            self._per_task_fast[task_id] = normalized * self.hypers.ema_timescale + self._per_task_fast[task_id] * (
                1.0 - self.hypers.ema_timescale
            )
            # Slow EMA (5x slower)
            slow_ts = self.hypers.ema_timescale * 0.2
            self._per_task_slow[task_id] = normalized * slow_ts + self._per_task_slow[task_id] * (1.0 - slow_ts)

        self._stale_dist = True
        self._invalidate_task_cache(task_id)

    def _update_basic_ema(self, task_id: int, score: float) -> None:
        """Update basic EMA tracking for a task with new score."""
        if task_id not in self._task_emas:
            self._task_emas[task_id] = (score, score * score, 1)
        else:
            ema_score, ema_squared, num_samples = self._task_emas[task_id]

            # Update EMAs
            alpha = min(1.0, self.hypers.ema_timescale * num_samples)
            new_ema_score = (1 - alpha) * ema_score + alpha * score
            new_ema_squared = (1 - alpha) * ema_squared + alpha * (score * score)

            self._task_emas[task_id] = (new_ema_score, new_ema_squared, num_samples + 1)

        # Invalidate cache for this task when EMA is updated
        self._invalidate_task_cache(task_id)

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
        """Get detailed stats including slice distribution analysis.

        Note: Learning progress stats are always included in stats() regardless of
        enable_detailed_logging, so they are not included here to avoid duplication.
        """
        # Only return slice analyzer stats (LP stats are always included in stats())
        return super().get_detailed_stats()  # Gets slice analyzer stats

    def _get_bidirectional_detailed_stats(self) -> Dict[str, float]:
        """Get detailed bidirectional learning progress statistics."""
        if not self._outcomes:
            return {
                "num_tracked_tasks": 0.0,
                "mean_task_success_rate": 0.0,
                "mean_learning_progress": 0.0,
                "num_active_tasks": 0.0,
            }

        learning_progress_array = self._learning_progress()
        mean_learning_progress = float(np.mean(learning_progress_array)) if len(learning_progress_array) > 0 else 0.0

        # Ensure task distribution is calculated for distribution stats
        if self._task_dist is None or self._stale_dist:
            self._calculate_task_distribution()

        stats = {
            "num_tracked_tasks": float(len(self._outcomes)),
            "mean_task_success_rate": float(np.mean(self._task_success_rate))
            if len(self._task_success_rate) > 0
            else 0.0,
            "mean_learning_progress": mean_learning_progress,
        }

        if self._task_dist is not None and len(self._task_dist) > 0:
            stats.update(
                {
                    "mean_sample_prob": float(np.mean(self._task_dist)),
                    "num_zeros_lp_dist": float(np.sum(self._task_dist == 0)),
                }
            )
        else:
            stats.update(
                {
                    "mean_sample_prob": 0.0,
                    "num_zeros_lp_dist": 0.0,
                }
            )

        return stats

    def _get_basic_detailed_stats(self) -> Dict[str, float]:
        """Get detailed basic learning progress statistics."""
        if not self._task_emas:
            return {
                "num_tracked_tasks": 0.0,
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

    # Bidirectional learning progress implementation (integrated from modules)

    def _update_bidirectional_progress(self):
        """Sync numpy arrays from per-task EMAs for distribution calculation and stats.

        Note: This no longer recalculates EMAs - it only syncs arrays from per-task dictionaries
        for compatibility with distribution calculation and statistics.
        """
        if not self._outcomes:
            return

        # Get all tracked task IDs
        task_ids = sorted(self._outcomes.keys())
        num_tasks = len(task_ids)

        if num_tasks == 0:
            return

        # Calculate task success rates for stats
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
            self._random_baseline = np.full(num_tasks, 0.5)

        # Create update mask for tasks with sufficient data
        self._update_mask = np.array([len(self._outcomes[task_id]) >= 2 for task_id in task_ids])

        # Sync arrays from per-task EMAs (for distribution calculation)
        if self._p_fast is None or len(self._p_fast) != num_tasks:
            self._p_fast = np.zeros(num_tasks)
            self._p_slow = np.zeros(num_tasks)
            self._p_true = np.zeros(num_tasks)

        # Ensure arrays are initialized (type checker guard)
        assert self._p_fast is not None and self._p_slow is not None and self._p_true is not None

        # Sync per-task EMAs to arrays
        for idx, task_id in enumerate(task_ids):
            if task_id in self._per_task_fast and task_id in self._per_task_slow:
                self._p_fast[idx] = self._per_task_fast[task_id]
                self._p_slow[idx] = self._per_task_slow[task_id]
            else:
                # Initialize from normalized success rate if not in per-task dicts
                baseline = 0.5
                denominator = max(1.0 - baseline, 0.01)
                normalized = (task_success_rates[idx] - baseline) / denominator
                self._p_fast[idx] = normalized
                self._p_slow[idx] = normalized
            self._p_true[idx] = task_success_rates[idx]

        self._task_success_rate = task_success_rates
        self._stale_dist = True

    def _learning_progress(self, reweight: bool = True) -> np.ndarray:
        """Calculate learning progress as the difference between fast and slow moving averages.

        Note: This builds arrays from per-task EMAs for distribution calculation.
        Individual task scores use per-task EMAs directly.
        """
        # Ensure arrays are synced from per-task EMAs
        if not self._outcomes:
            return np.array([])

        task_ids = sorted(self._outcomes.keys())
        if not task_ids:
            return np.array([])

        # Sync arrays if needed
        if self._p_fast is None or len(self._p_fast) != len(task_ids):
            self._update_bidirectional_progress()

        if self._p_fast is None or self._p_slow is None:
            return np.array([])

        task_ids = sorted(self._outcomes.keys())
        if not task_ids:
            return np.array([])

        fast_list: list[float] = []
        slow_list: list[float] = []
        for task_id in task_ids:
            fast = self._per_task_fast.get(task_id)
            slow = self._per_task_slow.get(task_id)

            if fast is None or slow is None:
                success_vals = self._outcomes.get(task_id, [])
                success_rate = np.mean(success_vals) if success_vals else DEFAULT_SUCCESS_RATE
                baseline = 0.5
                denominator = max(1.0 - baseline, 0.01)
                normalized = (success_rate - baseline) / denominator
                fast = slow = normalized

            fast_list.append(fast)
            slow_list.append(slow)

        fast_arr = np.asarray(fast_list, dtype=float)
        slow_arr = np.asarray(slow_list, dtype=float)

        if reweight:
            fast_arr = self._reweight(fast_arr)
            slow_arr = self._reweight(slow_arr)

        lp = np.abs(fast_arr - slow_arr)
        performance_bonus = np.maximum(fast_arr, 0) * 0.1

        return lp + performance_bonus

    def _reweight(self, probs: np.ndarray | float) -> np.ndarray | float:
        """Apply progress smoothing reweighting to probability values.

        Accepts either a scalar or an array and returns the same shape/type.
        """
        arr = np.asarray(probs, dtype=float)
        smoothing = self.hypers.progress_smoothing
        numerator = arr * (1.0 - smoothing)
        denominator = arr + smoothing * (1.0 - 2.0 * arr)

        # Prevent divide-by-zero or sign flips; mirror distribution behavior.
        denominator = np.where(denominator <= 0, 1.0, denominator)
        result = numerator / denominator

        if np.ndim(probs) == 0:
            return float(result)
        return result

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid function to array values with clipping for stability."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _normalize_bidirectional_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """Apply exploration floor, center, sigmoid, and normalize."""
        if raw_scores.size == 0:
            return raw_scores

        # Progress smoothing nudges scores toward their mean to damp extremes
        if self.hypers.progress_smoothing > 0:
            mean_score = float(np.mean(raw_scores))
            smoothing = self.hypers.progress_smoothing
            raw_scores = raw_scores * (1.0 - smoothing) + mean_score * smoothing

        # Ensure every task retains some exploration weight so it can still be sampled
        min_weight = max(self.hypers.exploration_bonus, 1e-6)
        raw_scores = np.maximum(raw_scores, min_weight)

        # Center (but do not standardize) so smoothing magnitude is preserved
        if len(raw_scores) > 1:
            raw_scores = raw_scores - np.mean(raw_scores)

        subprobs = self._sigmoid(raw_scores)

        total = float(np.sum(subprobs))
        if total > 0:
            return subprobs / total
        return np.ones_like(subprobs) / len(subprobs)

    def _calculate_task_distribution(self) -> None:
        """Compute a normalized sampling distribution for reporting stats."""
        if not self._outcomes:
            self._task_dist = None
            self._task_success_rate = np.array([])
            self._stale_dist = False
            return

        task_ids = sorted(self._outcomes.keys())

        # Success rates for stats (not used for sampling)
        self._task_success_rate = np.array(
            [np.mean(self._outcomes[tid]) if self._outcomes[tid] else DEFAULT_SUCCESS_RATE for tid in task_ids],
            dtype=float,
        )

        lp = self._learning_progress(reweight=False)
        if lp.size == 0:
            self._task_dist = None
            self._stale_dist = False
            return

        # Ensure every task has some weight
        lp = np.maximum(lp, self.hypers.exploration_bonus)
        total = float(np.sum(lp))
        self._task_dist = lp / total if total > 0 else np.ones_like(lp) / len(lp)
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
                    # Deep-copy mutable state to avoid aliasing while training continues
                    "outcomes": {k: list(v) for k, v in self._outcomes.items()},
                    "counter": dict(self._counter),
                    "per_task_fast": dict(self._per_task_fast),
                    "per_task_slow": dict(self._per_task_slow),
                    "p_fast": self._p_fast.tolist() if self._p_fast is not None else None,
                    "p_slow": self._p_slow.tolist() if self._p_slow is not None else None,
                    "p_true": self._p_true.tolist() if self._p_true is not None else None,
                    "random_baseline": self._random_baseline.tolist() if self._random_baseline is not None else None,
                    "task_success_rate": self._task_success_rate.tolist(),
                    "update_mask": self._update_mask.tolist(),
                    "sample_levels": self._sample_levels.tolist(),
                    "task_dist": self._task_dist.tolist() if self._task_dist is not None else None,
                    "stale_dist": bool(self._stale_dist),
                    "score_cache": dict(self._score_cache),
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
            self._outcomes = state.get("outcomes", {})
            self._counter = state.get("counter", {})

            # Restore per-task EMAs (new format)
            if "per_task_fast" in state and "per_task_slow" in state:
                self._per_task_fast = state.get("per_task_fast", {})
                self._per_task_slow = state.get("per_task_slow", {})
            else:
                # Backward compatibility: reconstruct per-task EMAs from arrays
                self._per_task_fast = {}
                self._per_task_slow = {}
                p_fast_arr = state.get("p_fast")
                p_slow_arr = state.get("p_slow")
                if p_fast_arr is not None and p_slow_arr is not None:
                    task_ids = sorted(self._outcomes.keys())
                    p_fast = np.array(p_fast_arr)
                    p_slow = np.array(p_slow_arr)
                    for idx, task_id in enumerate(task_ids):
                        if idx < len(p_fast) and idx < len(p_slow):
                            self._per_task_fast[task_id] = float(p_fast[idx])
                            self._per_task_slow[task_id] = float(p_slow[idx])

            self._p_fast = np.array(state.get("p_fast")) if state.get("p_fast") is not None else None
            self._p_slow = np.array(state.get("p_slow")) if state.get("p_slow") is not None else None
            self._p_true = np.array(state.get("p_true")) if state.get("p_true") is not None else None
            self._random_baseline = (
                np.array(state.get("random_baseline")) if state.get("random_baseline") is not None else None
            )
            self._task_success_rate = np.array(state.get("task_success_rate", []))
            self._update_mask = np.array(state.get("update_mask", []))
            self._sample_levels = np.array(state.get("sample_levels", []))
            self._task_dist = np.array(state.get("task_dist")) if state.get("task_dist") is not None else None
            self._stale_dist = bool(state.get("stale_dist", True))
            self._score_cache = dict(state.get("score_cache", {}))
            self._cache_valid_tasks = set(state.get("cache_valid_tasks", []))
