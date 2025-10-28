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
        self._baseline_initialized: Optional[np.ndarray] = None  # Track which baselines are set
        self._task_success_rate: np.ndarray = np.array([])
        self._update_mask: np.ndarray = np.array([])
        self._sample_levels: np.ndarray = np.array([])

        # Cache for task distribution and scores
        self._task_dist: Optional[np.ndarray] = None
        self._raw_lp_scores: Optional[np.ndarray] = None  # Pre-zscore LP scores (after smoothing/reweighting)
        self._postzscored_lp_scores: Optional[np.ndarray] = None  # Post-zscore LP scores (before sigmoid)
        self._stale_dist = True
        self._score_cache: Dict[int, float] = {}
        self._raw_lp_cache: Dict[int, float] = {}  # Cache for raw LP scores
        self._postzscored_lp_cache: Dict[int, float] = {}  # Cache for post-zscore LP scores
        self._cache_valid_tasks: set[int] = set()

        # Track first 3 tasks for detailed wandb metrics
        self._tracked_task_ids: Dict[int, int] = {}  # Maps task_id -> position (0, 1, 2)
        self._tracked_task_metrics: Dict[int, Dict[str, float]] = {}  # Store latest metrics for each tracked task

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

                    # Store metrics for tracked tasks (for wandb logging)
                    if task_id in self._tracked_task_ids:
                        raw_lp = self._learning_progress()[task_idx] if len(self._learning_progress()) > 0 else 0.0
                        self._tracked_task_metrics[task_id] = {
                            "raw_lp": raw_lp,
                            "final_score": score,
                        }
                else:
                    score = self.config.exploration_bonus
            else:
                score = self.config.exploration_bonus

        # Cache the computed score
        self._score_cache[task_id] = score
        self._cache_valid_tasks.add(task_id)
        return score

    def get_raw_lp_score(self, task_id: int, tracker: TaskTracker) -> float:
        """Get raw LP score before z-score normalization (but after smoothing/reweighting).

        This returns the LP value after applying smoothing, performance bonus, and reweighting,
        but before z-score normalization and sigmoid transformation.

        Args:
            task_id: ID of task to score
            tracker: TaskTracker instance

        Returns:
            Raw LP score before z-score transformation
        """
        # Return cached raw LP if valid
        if task_id in self._cache_valid_tasks and task_id in self._raw_lp_cache:
            return self._raw_lp_cache[task_id]

        task_stats = tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < 2:
            # New tasks get exploration bonus
            raw_lp = self.config.exploration_bonus
        elif task_id not in self._outcomes or len(self._outcomes[task_id]) < 2:
            # Tasks without sufficient data get exploration bonus
            raw_lp = self.config.exploration_bonus
        else:
            # Calculate bidirectional learning progress
            self._update_bidirectional_progress()

            # Get raw LP scores if needed
            if self._raw_lp_scores is None or self._stale_dist:
                self._calculate_task_distribution()

            # Find task index in our tracking
            task_indices = list(self._outcomes.keys())
            if task_id in task_indices and self._raw_lp_scores is not None:
                task_idx = task_indices.index(task_id)
                if task_idx < len(self._raw_lp_scores):
                    raw_lp = float(self._raw_lp_scores[task_idx])
                else:
                    raw_lp = self.config.exploration_bonus
            else:
                raw_lp = self.config.exploration_bonus

        # Cache the computed raw LP
        self._raw_lp_cache[task_id] = raw_lp
        return raw_lp

    def get_postzscored_lp_score(self, task_id: int, tracker: TaskTracker) -> float:
        """Get LP score after z-score normalization but before sigmoid.

        This returns the LP value after z-score normalization (or temperature scaling)
        but before sigmoid transformation and final normalization.

        Args:
            task_id: ID of task to score
            tracker: TaskTracker instance

        Returns:
            Post-z-score LP score before sigmoid
        """
        # Return cached post-zscore LP if valid
        if task_id in self._cache_valid_tasks and task_id in self._postzscored_lp_cache:
            return self._postzscored_lp_cache[task_id]

        task_stats = tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < 2:
            # New tasks get exploration bonus (not z-scored)
            postzscored_lp = self.config.exploration_bonus
        elif task_id not in self._outcomes or len(self._outcomes[task_id]) < 2:
            # Tasks without sufficient data get exploration bonus
            postzscored_lp = self.config.exploration_bonus
        else:
            # Calculate bidirectional learning progress
            self._update_bidirectional_progress()

            # Get post-zscore LP scores if needed
            if self._postzscored_lp_scores is None or self._stale_dist:
                self._calculate_task_distribution()

            # Find task index in our tracking
            task_indices = list(self._outcomes.keys())
            if task_id in task_indices and self._postzscored_lp_scores is not None:
                task_idx = task_indices.index(task_id)
                if task_idx < len(self._postzscored_lp_scores):
                    postzscored_lp = float(self._postzscored_lp_scores[task_idx])
                else:
                    postzscored_lp = self.config.exploration_bonus
            else:
                postzscored_lp = self.config.exploration_bonus

        # Cache the computed post-zscore LP
        self._postzscored_lp_cache[task_id] = postzscored_lp
        return postzscored_lp

    def update_with_score(self, task_id: int, score: float) -> None:
        """Update bidirectional EMA tracking for a task with new score."""
        # Convert score to success rate (assuming score is between 0 and 1)
        success_rate = score  # max(0.0, min(1.0, score))

        # Track first 3 unique tasks for detailed wandb metrics
        if task_id not in self._tracked_task_ids and len(self._tracked_task_ids) < 3:
            next_position = len(self._tracked_task_ids)
            self._tracked_task_ids[task_id] = next_position
            self._tracked_task_metrics[task_id] = {}

        # Initialize outcomes for new tasks
        if task_id not in self._outcomes:
            self._outcomes[task_id] = []

        # Add outcome and maintain memory limit
        self._outcomes[task_id].append(success_rate)
        self._outcomes[task_id] = self._outcomes[task_id][-self.config.memory :]

        # Store raw reward for tracked tasks (for wandb)
        if task_id in self._tracked_task_ids:
            self._tracked_task_metrics[task_id].update(
                {
                    "raw_reward": score,
                    "clamped_reward": success_rate,
                    "sample_count": len(self._outcomes[task_id]),
                }
            )

        # Update bidirectional progress to ensure EMAs are updated
        self._update_bidirectional_progress()

        # Mark distribution as stale
        self._stale_dist = True
        self._cache_valid_tasks.discard(task_id)

    def remove_task(self, task_id: int) -> None:
        """Remove task from scoring system."""
        self._outcomes.pop(task_id, None)
        self._score_cache.pop(task_id, None)
        self._raw_lp_cache.pop(task_id, None)
        self._postzscored_lp_cache.pop(task_id, None)
        self._cache_valid_tasks.discard(task_id)
        self._stale_dist = True

    def get_stats(self) -> Dict[str, float]:
        """Get detailed bidirectional learning progress statistics."""
        if not self._outcomes:
            return {
                "mean_task_success_rate": 0.0,
                "mean_learning_progress": 0.0,
                "mean_lp_score": 0.0,
            }

        self._update_bidirectional_progress()

        stats = {
            "mean_task_success_rate": float(np.mean(self._task_success_rate))
            if len(self._task_success_rate) > 0
            else 0.0,
        }

        # Add mean LP score from score cache
        if self._score_cache:
            stats["mean_lp_score"] = float(np.mean(list(self._score_cache.values())))
        else:
            stats["mean_lp_score"] = 0.0

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

        # Add detailed metrics for tracked tasks (first 3 tasks) if troubleshooting logging is enabled
        if self.config.show_curriculum_troubleshooting_logging:
            for task_id, position in self._tracked_task_ids.items():
                if task_id in self._tracked_task_metrics:
                    metrics = self._tracked_task_metrics[task_id]
                    for metric_name, value in metrics.items():
                        stats[f"tracked_task_{position}/{metric_name}"] = float(value)

        return stats

    def get_state(self) -> Dict[str, Any]:
        """Serialize bidirectional scorer state."""
        return {
            "outcomes": {k: v for k, v in self._outcomes.items()},
            "p_fast": self._p_fast.tolist() if self._p_fast is not None else None,
            "p_slow": self._p_slow.tolist() if self._p_slow is not None else None,
            "p_true": self._p_true.tolist() if self._p_true is not None else None,
            "random_baseline": self._random_baseline.tolist() if self._random_baseline is not None else None,
            "baseline_initialized": self._baseline_initialized.tolist()
            if self._baseline_initialized is not None
            else None,
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
        self._baseline_initialized = (
            np.array(state["baseline_initialized"], dtype=bool)
            if state.get("baseline_initialized") is not None
            else None
        )
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
        self._raw_lp_cache.clear()
        self._postzscored_lp_cache.clear()

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
            # Initialize random baseline array if needed
            if self._random_baseline is None or len(self._random_baseline) != num_tasks:
                # Initialize baseline array to zeros
                self._random_baseline = np.zeros(num_tasks)
                # Track which tasks have had their baseline set
                self._baseline_initialized = np.zeros(num_tasks, dtype=bool)

            # Ensure _baseline_initialized is properly initialized (for resize case)
            if self._baseline_initialized is None or len(self._baseline_initialized) != num_tasks:
                self._baseline_initialized = np.zeros(num_tasks, dtype=bool)

            # Set baseline for new tasks (first observation, capped at 0.75)
            # This captures the "floor" performance - the starting-point skill level
            new_tasks_mask = self._update_mask & ~self._baseline_initialized
            if np.any(new_tasks_mask):
                # Capture FIRST observation as baseline, capped at 0.75 to prevent division by zero
                # and ensure there's room for improvement (1.0 - B_i > 0)
                # Use the first outcome value, not the current TSR (which is the mean)
                for i, task_id in enumerate(task_ids):
                    if new_tasks_mask[i] and task_id in self._outcomes and len(self._outcomes[task_id]) > 0:
                        first_observation = self._outcomes[task_id][0]
                        self._random_baseline[i] = min(first_observation, 0.75)
                self._baseline_initialized[new_tasks_mask] = True

            # Calculate normalized "mastery" score: p_i = (TSR_i - B_i) / (1.0 - B_i)
            # This measures progress from the baseline to perfect performance
            improvement_over_baseline = np.maximum(
                task_success_rates[self._update_mask] - self._random_baseline[self._update_mask],
                0.0,
            )

            total_possible_improvement = 1.0 - self._random_baseline[self._update_mask]
            # Handle edge case where baseline is 1.0 (shouldn't happen with 0.75 cap, but be safe)
            total_possible_improvement = np.where(total_possible_improvement <= 1e-10, 1.0, total_possible_improvement)

            # This is the "mastery" score p_i that feeds into the EMAs
            normalized_task_success_rates = improvement_over_baseline / total_possible_improvement
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

                # Store EMA values for tracked tasks (for wandb)
                for task_id in self._tracked_task_ids:
                    if task_id in task_ids and task_id in self._tracked_task_metrics:
                        task_idx = task_ids.index(task_id)
                        if task_idx < len(self._p_fast):
                            mean_reward = task_success_rates[task_idx]
                            fast_ema = self._p_fast[task_idx]
                            slow_ema = self._p_slow[task_idx]
                            lp = fast_ema - slow_ema
                            self._tracked_task_metrics[task_id].update(
                                {
                                    "mean_reward": mean_reward,
                                    "fast_ema": fast_ema,
                                    "slow_ema": slow_ema,
                                    "raw_lp": lp,
                                }
                            )

        self._task_success_rate = task_success_rates
        self._stale_dist = True

    def _reweight(self, p: float) -> float:
        """Reweight performance signal to amplify unsolved/solved task signals.

        Applies the reweighting function: R(p) = p * (1 - theta) / (p + theta * (1 - 2p))

        Args:
            p: Performance value (success rate) between 0 and 1

        Returns:
            Reweighted performance value

        Notes:
            - When theta = 0.5, R(p) ≈ p (effectively OFF)
            - When theta is low (e.g., 0.05), amplifies signal from unsolved tasks (p~0)
              and dampens signal from partially-solved tasks (p~0.5)
            - Higher theta values reweight toward higher performance tasks
        """
        theta = self.config.early_progress_amplification

        # Numerator: p * (1 - theta)
        numerator = p * (1.0 - theta)

        # Denominator: p + theta * (1 - 2p)
        denominator = p + theta * (1.0 - 2.0 * p)

        # Handle potential division by zero
        # If p=0 and theta=0, or if denominator is very small
        if abs(denominator) < 1e-10:
            # When p=0, R(p) should be 0; when p=1, R(p) should be 1
            return 0.0 if p < 0.5 else 1.0

        return numerator / denominator

    def _learning_progress(self) -> np.ndarray:
        """Calculate raw learning progress with optional reweighting.

        Applies reweighting to fast and slow EMAs before computing absolute difference.
        This amplifies learning signal from unsolved tasks when early_progress_amplification
        is set to a low value (e.g., 0.05).
        """
        if self._p_fast is None or self._p_slow is None:
            return np.array([])

        # Apply reweighting if not at default (0.5)
        if abs(self.config.early_progress_amplification - 0.5) > 1e-6:
            # Apply reweighting element-wise to both fast and slow EMAs
            # Need to clip to [0, 1] range to ensure valid probability values
            p_fast_clipped = np.clip(self._p_fast, 0.0, 1.0)
            p_slow_clipped = np.clip(self._p_slow, 0.0, 1.0)

            # Vectorize the reweight function and apply
            reweighted_fast = np.vectorize(self._reweight)(p_fast_clipped)
            reweighted_slow = np.vectorize(self._reweight)(p_slow_clipped)

            # Calculate LP from reweighted signals
            return np.abs(reweighted_fast - reweighted_slow)
        else:
            # Default behavior: unweighted LP
            return np.abs(self._p_fast - self._p_slow)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function with clipping to prevent overflow."""
        x_clipped = np.clip(x, -500, 500)  # Clip to prevent overflow
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def _calculate_task_distribution(self) -> None:
        """Calculate task sampling distribution from learning progress."""
        if not self._outcomes:
            self._task_dist = None
            self._raw_lp_scores = None
            self._postzscored_lp_scores = None
            self._stale_dist = False
            return

        learning_progress = self._learning_progress()
        if len(learning_progress) == 0:
            self._task_dist = None
            self._raw_lp_scores = None
            self._postzscored_lp_scores = None
            self._stale_dist = False
            return

        # Apply smoothing to all tasks (even those with negative/zero learning progress)
        # This ensures all tasks get non-zero probability
        subprobs = learning_progress + self.config.progress_smoothing

        # Add performance bonus if configured
        if self.config.performance_bonus_weight > 0 and self._p_true is not None:
            performance_bonus = self._p_true * self.config.performance_bonus_weight
            subprobs = subprobs + performance_bonus

        # Store raw LP scores (before z-score normalization)
        self._raw_lp_scores = subprobs.copy().astype(np.float32)

        # Apply temperature scaling or z-score normalization before sigmoid
        # Temperature controls how LP scores are transformed before sigmoid:
        # - temp > 0: Divide by temperature (low temp amplifies differences)
        # - temp = 0: Z-score normalize (standardize to mean=0, std=1)
        temperature = self.config.lp_score_temperature
        if temperature == 0:
            # Z-score normalization: center at mean and normalize by std
            # This makes sigmoid operate on standardized scores, preventing saturation
            mean = np.mean(subprobs)
            std = np.std(subprobs)
            if std > 1e-10:  # Avoid division by zero
                subprobs = (subprobs - mean) / std
            # else: if std is zero, all tasks have identical LP, leave as-is
        elif temperature > 0:
            # Temperature scaling: divide by temperature
            # Low temp (< 1) amplifies differences, high temp (> 1) smooths them
            subprobs = subprobs / temperature
        # else: negative temperature is invalid, leave subprobs unchanged

        # Store post-z-score LP scores (after z-score/temperature, before sigmoid)
        self._postzscored_lp_scores = subprobs.copy().astype(np.float32)

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
        stats = {
            "mean_learning_progress": 0.0,  # Could compute from all tasks if needed
            "num_cached_scores": len(self._score_cache),
        }

        # Add mean LP score from score cache
        if self._score_cache:
            stats["mean_lp_score"] = float(np.mean(list(self._score_cache.values())))
        else:
            stats["mean_lp_score"] = 0.0

        return stats

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
