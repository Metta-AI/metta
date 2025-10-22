"""Learning Progress Scorer implementations.

Provides strategy pattern for different LP scoring algorithms:
- BidirectionalLPScorer: Fast/slow EMA-based learning progress
- BasicLPScorer: Variance-based learning progress
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from .task_tracker import TaskTracker

if TYPE_CHECKING:
    from .learning_progress_algorithm import LearningProgressConfig

# Constants for bidirectional learning progress
DEFAULT_SUCCESS_RATE = 0.0


class LPScorer(ABC):
    """Abstract base class for learning progress scoring strategies."""

    def __init__(self, config: "LearningProgressConfig"):
        """Initialize scorer with configuration.

        Args:
            config: Learning progress configuration object
        """
        self.config = config

    @abstractmethod
    def score_task(self, task_id: int, tracker: TaskTracker) -> float:
        """Calculate learning progress score for a task.

        Args:
            task_id: ID of task to score
            tracker: TaskTracker instance to read performance data from

        Returns:
            Learning progress score (higher = more learning potential)
        """
        ...

    @abstractmethod
    def update_with_score(self, task_id: int, score: float) -> None:
        """Update internal scorer state with new task performance.

        Args:
            task_id: ID of task that was completed
            score: Performance score from task completion
        """
        ...

    @abstractmethod
    def remove_task(self, task_id: int) -> None:
        """Clean up task-specific state when task is evicted.

        Args:
            task_id: ID of task being removed
        """
        ...

    @abstractmethod
    def get_stats(self) -> Dict[str, float]:
        """Get scorer-specific statistics for monitoring.

        Returns:
            Dictionary of stat_name -> value
        """
        ...

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Serialize scorer state for checkpointing.

        Returns:
            Dictionary of state data
        """
        ...

    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """Deserialize scorer state from checkpoint.

        Args:
            state: Dictionary of state data to restore
        """
        ...

    @abstractmethod
    def invalidate_cache(self) -> None:
        """Invalidate any internal caches."""
        ...


class BidirectionalLPScorer(LPScorer):
    """Bidirectional learning progress using fast/slow exponential moving averages.

    This scorer tracks recent task outcomes and computes learning progress
    by comparing fast and slow EMAs of performance. Tasks with positive
    learning progress (fast EMA > slow EMA) are prioritized.
    """

    def __init__(self, config: "LearningProgressConfig"):
        """Initialize bidirectional scorer.

        Args:
            config: Learning progress configuration
        """
        super().__init__(config)

        # Bidirectional learning progress tracking
        self._outcomes: Dict[int, List[float]] = {}
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
        self._score_cache: Dict[int, float] = {}
        self._cache_valid_tasks: set[int] = set()

    def score_task(self, task_id: int, tracker: TaskTracker) -> float:
        """Calculate bidirectional learning progress score for a task."""
        # Return cached score if valid
        if task_id in self._cache_valid_tasks and task_id in self._score_cache:
            return self._score_cache[task_id]

        task_stats = tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < 2:
            # New tasks get exploration bonus
            score = self.config.exploration_bonus
        elif task_id not in self._outcomes or len(self._outcomes[task_id]) < 2:
            # Tasks without sufficient data get exploration bonus
            score = self.config.exploration_bonus
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
                    score = self.config.exploration_bonus
            else:
                score = self.config.exploration_bonus

        # Cache the computed score
        self._score_cache[task_id] = score
        self._cache_valid_tasks.add(task_id)
        return score

    def update_with_score(self, task_id: int, score: float) -> None:
        """Update bidirectional EMA tracking for a task with new score."""
        # Convert score to success rate (assuming score is between 0 and 1)
        success_rate = max(0.0, min(1.0, score))

        # Initialize outcomes for new tasks
        if task_id not in self._outcomes:
            self._outcomes[task_id] = []

        # Add outcome and maintain memory limit
        self._outcomes[task_id].append(success_rate)
        self._outcomes[task_id] = self._outcomes[task_id][-self.config.memory :]

        # Update bidirectional progress to ensure EMAs are updated
        self._update_bidirectional_progress()

        # Mark distribution as stale
        self._stale_dist = True
        self._cache_valid_tasks.discard(task_id)

    def remove_task(self, task_id: int) -> None:
        """Remove task from scoring system."""
        self._outcomes.pop(task_id, None)
        self._score_cache.pop(task_id, None)
        self._cache_valid_tasks.discard(task_id)
        self._stale_dist = True

    def get_stats(self) -> Dict[str, float]:
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

    def get_state(self) -> Dict[str, Any]:
        """Serialize bidirectional scorer state."""
        return {
            "outcomes": {k: v for k, v in self._outcomes.items()},
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

    def load_state(self, state: Dict[str, Any]) -> None:
        """Deserialize bidirectional scorer state."""
        self._outcomes = {int(k): v for k, v in state.get("outcomes", {}).items()}
        self._p_fast = np.array(state["p_fast"]) if state.get("p_fast") is not None else None
        self._p_slow = np.array(state["p_slow"]) if state.get("p_slow") is not None else None
        self._p_true = np.array(state["p_true"]) if state.get("p_true") is not None else None
        self._random_baseline = np.array(state["random_baseline"]) if state.get("random_baseline") is not None else None
        self._task_success_rate = np.array(state.get("task_success_rate", []))
        self._update_mask = np.array(state.get("update_mask", []))
        self._sample_levels = np.array(state.get("sample_levels", []))
        self._task_dist = np.array(state["task_dist"]) if state.get("task_dist") is not None else None
        self._stale_dist = state.get("stale_dist", True)
        self._score_cache = state.get("score_cache", {})
        self._cache_valid_tasks = set(state.get("cache_valid_tasks", []))

    def invalidate_cache(self) -> None:
        """Invalidate score cache."""
        self._cache_valid_tasks.clear()
        self._score_cache.clear()

    def _update_bidirectional_progress(self) -> None:
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

        # Create update mask for tasks with sufficient data
        self._update_mask = np.array([len(self._outcomes[task_id]) >= 2 for task_id in task_ids])

        if not np.any(self._update_mask):
            return

        # Optionally normalize by random baseline
        if self.config.use_baseline_normalization:
            # Initialize random baseline if needed
            if self._random_baseline is None or len(self._random_baseline) != num_tasks:
                # Random baseline should represent baseline/random performance, typically around 0.5
                # Ideally, we would find this value out on a task by task level.
                self._random_baseline = np.full(num_tasks, 0.5)

            # Handle division by zero in normalization
            denominator = 1.0 - self._random_baseline[self._update_mask]
            denominator = np.where(denominator <= 0, 1.0, denominator)

            # Normalize by baseline to make LP comparable across different task difficulties
            normalized_task_success_rates = (
                task_success_rates[self._update_mask] - self._random_baseline[self._update_mask]
            ) / denominator
        else:
            # Use raw success rates directly (default)
            # Learning progress = rate of change in raw performance
            normalized_task_success_rates = task_success_rates[self._update_mask]

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

            # Update EMAs
            if self._p_fast is not None and self._p_slow is not None and self._p_true is not None:
                # Fast EMA uses the configured timescale
                self._p_fast[self._update_mask] = (
                    normalized_task_success_rates * self.config.ema_timescale
                    + self._p_fast[self._update_mask] * (1.0 - self.config.ema_timescale)
                )
                # Slow EMA uses a much slower timescale
                slow_timescale = self.config.ema_timescale * self.config.slow_timescale_factor
                self._p_slow[self._update_mask] = normalized_task_success_rates * slow_timescale + self._p_slow[
                    self._update_mask
                ] * (1.0 - slow_timescale)
                self._p_true[self._update_mask] = task_success_rates[
                    self._update_mask
                ] * self.config.ema_timescale + self._p_true[self._update_mask] * (1.0 - self.config.ema_timescale)

        self._task_success_rate = task_success_rates
        self._stale_dist = True

    def _learning_progress(self) -> np.ndarray:
        """Calculate raw learning progress (fast EMA - slow EMA)."""
        if self._p_fast is None or self._p_slow is None:
            return np.array([])
        return self._p_fast - self._p_slow

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function with clipping to prevent overflow."""
        x_clipped = np.clip(x, -500, 500)  # Clip to prevent overflow
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def _calculate_task_distribution(self) -> None:
        """Calculate task sampling distribution from learning progress."""
        if not self._outcomes:
            self._task_dist = None
            self._stale_dist = False
            return

        learning_progress = self._learning_progress()
        if len(learning_progress) == 0:
            self._task_dist = None
            self._stale_dist = False
            return

        # Apply smoothing to all tasks (even those with negative/zero learning progress)
        # This ensures all tasks get non-zero probability
        subprobs = learning_progress + self.config.progress_smoothing

        # Add performance bonus if configured
        if self.config.performance_bonus_weight > 0 and self._p_true is not None:
            performance_bonus = self._p_true * self.config.performance_bonus_weight
            subprobs = subprobs + performance_bonus

        subprobs = self._sigmoid(subprobs)

        # Normalize to sum to 1
        sum_probs = np.sum(subprobs)
        if sum_probs > 0:
            task_dist = subprobs / sum_probs
        else:
            task_dist = np.ones_like(subprobs) / len(subprobs)

        self._task_dist = task_dist.astype(np.float32)
        self._stale_dist = False


class BasicLPScorer(LPScorer):
    """Basic learning progress using variance estimation from EMAs.

    This scorer computes learning progress as the variance in task performance,
    estimated from first and second moment EMAs. Higher variance indicates
    more learning opportunity.
    """

    def __init__(self, config: "LearningProgressConfig"):
        """Initialize basic scorer.

        Args:
            config: Learning progress configuration
        """
        super().__init__(config)

        # Cache for scores
        self._score_cache: Dict[int, float] = {}
        self._cache_valid_tasks: set[int] = set()

    def score_task(self, task_id: int, tracker: TaskTracker) -> float:
        """Calculate basic learning progress score using EMA variance."""
        # Return cached score if valid
        if task_id in self._cache_valid_tasks and task_id in self._score_cache:
            return self._score_cache[task_id]

        task_stats = tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < 2:
            score = self.config.exploration_bonus
        else:
            # Use TaskTracker's reward_ema and ema_squared for variance calculation
            ema_score = task_stats["reward_ema"]
            ema_squared = task_stats["ema_squared"]
            completion_count = task_stats["completion_count"]

            # Calculate variance: Var(X) = E[X²] - (E[X])²
            variance = max(0.0, ema_squared - (ema_score * ema_score))

            # Calculate standard deviation
            std_dev = np.sqrt(variance)

            # Use exploration bonus for tasks with insufficient samples
            if completion_count < self.config.min_samples_for_lp:
                score = self.config.exploration_bonus
            else:
                # Learning progress is approximated by variance in performance
                score = std_dev

        # Cache the computed score
        self._score_cache[task_id] = score
        self._cache_valid_tasks.add(task_id)
        return score

    def update_with_score(self, task_id: int, score: float) -> None:
        """Update basic EMA tracking for a task with new score.

        Note: Basic mode relies primarily on TaskTracker's EMAs.
        """
        # Invalidate cache for this task
        self._cache_valid_tasks.discard(task_id)

    def remove_task(self, task_id: int) -> None:
        """Remove task from scoring system."""
        self._score_cache.pop(task_id, None)
        self._cache_valid_tasks.discard(task_id)

    def get_stats(self) -> Dict[str, float]:
        """Get detailed basic learning progress statistics."""
        # Basic mode stats are minimal since most data is in TaskTracker
        return {
            "mean_learning_progress": 0.0,  # Could compute from all tasks if needed
            "num_cached_scores": len(self._score_cache),
        }

    def get_state(self) -> Dict[str, Any]:
        """Serialize basic scorer state."""
        return {
            "score_cache": self._score_cache,
            "cache_valid_tasks": list(self._cache_valid_tasks),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Deserialize basic scorer state."""
        self._score_cache = state.get("score_cache", {})
        self._cache_valid_tasks = set(state.get("cache_valid_tasks", []))

    def invalidate_cache(self) -> None:
        """Invalidate score cache."""
        self._cache_valid_tasks.clear()
        self._score_cache.clear()
