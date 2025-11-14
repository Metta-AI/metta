"""Learning progress scoring strategies for curriculum algorithms.

This module implements the strategy pattern for different approaches to measuring learning
progress. Scorers analyze task performance data (stored in shared memory) and compute scores
that guide curriculum task selection - higher scores indicate better learning opportunities.

Key implementations:
- BidirectionalLPScorer: Compares fast/slow EMAs to detect performance changes (default)
- BasicLPScorer: Uses variance estimation from EMAs (legacy/simpler approach)
- LPScorer: Abstract base defining the scoring interface

Bidirectional LP intuition:
When fast EMA > slow EMA, the agent is improving (positive learning signal).
When fast EMA < slow EMA, the agent is regressing (still learning, just negatively).
The absolute difference |fast - slow| measures the rate of change.

All EMA state lives in shared memory (via TaskTracker), not in scorer instance variables.
This enables true multi-process learning progress tracking where all workers see the same
task performance history.

Why separate file: Scoring logic is complex and benefits from isolation. Multiple strategies
can coexist and be swapped at runtime via config, following the strategy pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from .task_tracker import TaskTracker

if TYPE_CHECKING:
    from .learning_progress_algorithm import LearningProgressConfig

# Constants for bidirectional learning progress
DEFAULT_SUCCESS_RATE = 0.0


class LPScorer(ABC):
    """Abstract base class for learning progress scoring strategies."""

    def __init__(self, config: "LearningProgressConfig", tracker: Optional[TaskTracker] = None):
        """Initialize scorer with configuration.

        Args:
            config: Learning progress configuration object
            tracker: Optional TaskTracker instance for shared memory access
        """
        self.config = config
        self.tracker = tracker

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

    def get_raw_lp_score(self, task_id: int, tracker: TaskTracker) -> float:
        """Get raw LP score before normalization (default: same as regular score).

        Args:
            task_id: ID of task to score
            tracker: TaskTracker instance

        Returns:
            Raw learning progress score
        """
        return self.score_task(task_id, tracker)

    def get_postzscored_lp_score(self, task_id: int, tracker: TaskTracker) -> float:
        """Get LP score after z-score but before sigmoid (default: same as regular score).

        Args:
            task_id: ID of task to score
            tracker: TaskTracker instance

        Returns:
            Post-z-scored learning progress score
        """
        return self.score_task(task_id, tracker)


class BidirectionalLPScorer(LPScorer):
    """Bidirectional learning progress using fast/slow exponential moving averages.

    This scorer tracks recent task outcomes and computes learning progress
    by comparing fast and slow EMAs of performance. Tasks with positive
    learning progress (fast EMA > slow EMA) are prioritized.
    """

    def __init__(self, config: "LearningProgressConfig", tracker: Optional[TaskTracker] = None):
        """Initialize bidirectional scorer.

        Args:
            config: Learning progress configuration
            tracker: TaskTracker instance for shared memory access (required for EMAs)
        """
        super().__init__(config, tracker)

        # Bidirectional learning progress tracking
        # All EMA state (p_fast, p_slow, p_true, random_baseline) is stored in shared memory (indices 13-16).
        # The scorer maintains only caches and working data, no persistent outcome buffers.

        # Track last outcome count to detect which tasks have new data (enables selective EMA updates)
        self._last_outcome_counts: Dict[int, int] = {}

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

    def _get_ema_from_shared_memory(self, task_id: int, tracker: TaskTracker) -> tuple[float, float, float, float]:
        """Get EMA values from shared memory for a task.

        Returns:
            Tuple of (p_fast, p_slow, p_true, random_baseline)
        """
        task_stats = tracker.get_task_stats(task_id)
        if task_stats is None:
            return 0.0, 0.0, 0.0, 0.0
        return (
            task_stats.get("p_fast", 0.0),
            task_stats.get("p_slow", 0.0),
            task_stats.get("p_true", 0.0),
            task_stats.get("random_baseline", 0.0),
        )

    def _write_ema_to_shared_memory(
        self, task_id: int, tracker: TaskTracker, p_fast: float, p_slow: float, p_true: float, random_baseline: float
    ) -> None:
        """Write EMA values to shared memory for a task."""
        if task_id not in tracker._task_id_to_index:
            return

        index = tracker._task_id_to_index[task_id]
        with tracker._backend.acquire_lock():
            task_data = tracker._backend.get_task_data(index)
            task_data[13] = p_fast
            task_data[14] = p_slow
            task_data[15] = p_true
            task_data[16] = random_baseline

    def score_task(self, task_id: int, tracker: TaskTracker) -> float:
        """Calculate bidirectional learning progress score for a task."""
        # Return cached score if valid
        if task_id in self._cache_valid_tasks and task_id in self._score_cache:
            return self._score_cache[task_id]

        # Check completion count from shared memory for sufficient data
        task_stats = tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < self.config.min_samples_for_lp:
            # Tasks without sufficient data get exploration bonus
            score = self.config.exploration_bonus
        else:
            # Calculate bidirectional learning progress
            self._update_bidirectional_progress(tracker)

            # Get task distribution if needed
            if self._task_dist is None or self._stale_dist:
                self._calculate_task_distribution()

            # Find task index in our tracking (must match insertion order from _update_bidirectional_progress)
            task_indices = tracker.get_all_tracked_tasks()
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

        # Cache the computed score and write to shared memory
        self._score_cache[task_id] = score
        self._cache_valid_tasks.add(task_id)
        tracker.update_lp_score(task_id, score)
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

        # Check completion count from shared memory for sufficient data
        task_stats = tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < self.config.min_samples_for_lp:
            # Tasks without sufficient data get exploration bonus
            raw_lp = self.config.exploration_bonus
        else:
            # Calculate bidirectional learning progress
            self._update_bidirectional_progress(tracker)

            # Get raw LP scores if needed
            if self._raw_lp_scores is None or self._stale_dist:
                self._calculate_task_distribution()

            # Find task index in our tracking (must match insertion order from _update_bidirectional_progress)
            task_indices = tracker.get_all_tracked_tasks()
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
        self._cache_valid_tasks.add(task_id)
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

        # Check completion count from shared memory for sufficient data
        task_stats = tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < self.config.min_samples_for_lp:
            # Tasks without sufficient data get exploration bonus
            postzscored_lp = self.config.exploration_bonus
        else:
            # Calculate bidirectional learning progress
            self._update_bidirectional_progress(tracker)

            # Get post-zscore LP scores if needed
            if self._postzscored_lp_scores is None or self._stale_dist:
                self._calculate_task_distribution()

            # Find task index in our tracking (must match insertion order from _update_bidirectional_progress)
            task_indices = tracker.get_all_tracked_tasks()
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
        self._cache_valid_tasks.add(task_id)
        return postzscored_lp

    def update_with_score(self, task_id: int, score: float) -> None:
        """Update bidirectional EMA tracking for a task with new score."""
        # Track first 3 unique tasks for detailed wandb metrics
        if task_id not in self._tracked_task_ids and len(self._tracked_task_ids) < 3:
            next_position = len(self._tracked_task_ids)
            self._tracked_task_ids[task_id] = next_position
            self._tracked_task_metrics[task_id] = {}

        # Store raw reward for tracked tasks (for wandb)
        if task_id in self._tracked_task_ids:
            self._tracked_task_metrics[task_id].update(
                {
                    "raw_reward": score,
                    "clamped_reward": score,
                }
            )

        # Update bidirectional progress for this specific task with the new score
        if self.tracker is not None:
            self._update_task_emas(task_id, score, self.tracker)

        # Mark distribution as stale
        self._stale_dist = True
        # Invalidate ALL caches because bidirectional LP depends on all tasks (due to z-score normalization)
        self._cache_valid_tasks.clear()
        self._score_cache.clear()
        self._raw_lp_cache.clear()
        self._postzscored_lp_cache.clear()

    def remove_task(self, task_id: int) -> None:
        """Remove task from scoring system."""
        self._score_cache.pop(task_id, None)
        self._raw_lp_cache.pop(task_id, None)
        self._postzscored_lp_cache.pop(task_id, None)
        self._cache_valid_tasks.discard(task_id)
        self._last_outcome_counts.pop(task_id, None)
        self._stale_dist = True

    def get_stats(self) -> Dict[str, float]:
        """Get detailed bidirectional learning progress statistics."""
        if self.tracker is None:
            return {
                "mean_task_success_rate": 0.0,
                "mean_learning_progress": 0.0,
                "mean_lp_score": 0.0,
            }

        task_ids = self.tracker.get_all_tracked_tasks()
        if not task_ids:
            return {
                "mean_task_success_rate": 0.0,
                "mean_learning_progress": 0.0,
                "mean_lp_score": 0.0,
            }

        self._update_bidirectional_progress(self.tracker)

        # Compute mean task success rate from shared memory success_rate_ema
        task_success_rates = []
        for task_id in task_ids:
            task_stats = self.tracker.get_task_stats(task_id)
            if task_stats and task_stats["completion_count"] > 0:
                task_success_rates.append(task_stats["success_rate_ema"])

        stats = {
            "mean_task_success_rate": float(np.mean(task_success_rates)) if task_success_rates else 0.0,
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
        """Serialize bidirectional scorer state.

        Note: EMAs (p_fast, p_slow, p_true, random_baseline) are stored in shared memory
        via the task tracker (indices 13-16), so they're not included here.
        Only scorer-specific working data (caches) is serialized.
        """
        return {
            "task_dist": self._task_dist.tolist() if self._task_dist is not None else None,
            "stale_dist": self._stale_dist,
            "score_cache": self._score_cache,
            "cache_valid_tasks": list(self._cache_valid_tasks),
            "last_outcome_counts": self._last_outcome_counts,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Deserialize bidirectional scorer state.

        Note: EMAs (p_fast, p_slow, p_true, random_baseline) are stored in shared memory
        via the task tracker (indices 13-16), so they're restored from there automatically.
        """
        self._task_dist = np.array(state["task_dist"]) if state.get("task_dist") is not None else None
        self._stale_dist = state.get("stale_dist", True)
        self._score_cache = state.get("score_cache", {})
        self._cache_valid_tasks = set(state.get("cache_valid_tasks", []))
        self._last_outcome_counts = {int(k): v for k, v in state.get("last_outcome_counts", {}).items()}

    def invalidate_cache(self) -> None:
        """Invalidate score cache."""
        self._cache_valid_tasks.clear()
        self._score_cache.clear()
        self._raw_lp_cache.clear()
        self._postzscored_lp_cache.clear()

    def _update_task_emas(self, task_id: int, score: float, tracker: TaskTracker) -> None:
        """Update bidirectional EMAs for a single task with a new score.

        Args:
            task_id: ID of the task to update
            score: New performance score for the task
            tracker: TaskTracker instance for shared memory access
        """
        # Get current EMA values from shared memory
        p_fast, p_slow, p_true, random_baseline = self._get_ema_from_shared_memory(task_id, tracker)

        task_success_rate = score

        # Handle baseline normalization if enabled
        if self.config.use_baseline_normalization:
            # Set baseline on first update (capped at 0.75)
            if random_baseline == 0.0:
                random_baseline = min(task_success_rate, 0.75)

            # Calculate normalized "mastery" score
            improvement_over_baseline = max(task_success_rate - random_baseline, 0.0)
            total_possible_improvement = max(1.0 - random_baseline, 1e-10)
            normalized_task_success_rate = improvement_over_baseline / total_possible_improvement
        else:
            # Use raw success rate
            normalized_task_success_rate = task_success_rate

        # Initialize or update EMAs
        if p_fast == 0.0 and p_slow == 0.0:
            # First update - initialize to current value
            p_fast = normalized_task_success_rate
            p_slow = normalized_task_success_rate
            p_true = task_success_rate
        else:
            # Update EMAs
            p_fast = normalized_task_success_rate * self.config.ema_timescale + p_fast * (
                1.0 - self.config.ema_timescale
            )
            slow_timescale = self.config.ema_timescale * self.config.slow_timescale_factor
            p_slow = normalized_task_success_rate * slow_timescale + p_slow * (1.0 - slow_timescale)
            p_true = task_success_rate * self.config.ema_timescale + p_true * (1.0 - self.config.ema_timescale)

        # Write updated EMAs back to shared memory
        self._write_ema_to_shared_memory(task_id, tracker, p_fast, p_slow, p_true, random_baseline)

        # Store EMA values for tracked tasks (for wandb)
        if task_id in self._tracked_task_ids and task_id in self._tracked_task_metrics:
            lp = p_fast - p_slow
            self._tracked_task_metrics[task_id].update(
                {
                    "mean_reward": task_success_rate,
                    "fast_ema": p_fast,
                    "slow_ema": p_slow,
                    "raw_lp": lp,
                }
            )

        # Mark distribution as stale since EMAs changed
        self._stale_dist = True

    def _update_bidirectional_progress(self, tracker: TaskTracker) -> None:
        """Update bidirectional learning progress for all tasks.

        This method updates EMAs for all tracked tasks that have new data since last update.
        Used when explicitly called (e.g., in tests) or when getting stats.
        """
        # Get all tracked task IDs from tracker
        task_ids = tracker.get_all_tracked_tasks()
        if not task_ids:
            return

        # Process each task individually
        for task_id in task_ids:
            # Get task stats from shared memory
            task_stats = tracker.get_task_stats(task_id)
            if not task_stats:
                continue

            completion_count = task_stats["completion_count"]

            # Check if task has new completions since last update
            has_new_data = completion_count > 0 and completion_count != self._last_outcome_counts.get(task_id, 0)

            if not has_new_data:
                continue

            # Use reward_ema as the current performance value for batch updates
            score = task_stats["reward_ema"]
            self._update_task_emas(task_id, score, tracker)

            # Update the last completion count AFTER processing EMAs
            self._last_outcome_counts[task_id] = int(completion_count)

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

        Now builds arrays from shared memory instead of using cached arrays.
        """
        if self.tracker is None:
            return np.array([])

        # Build arrays from shared memory
        task_ids = self.tracker.get_all_tracked_tasks()
        if not task_ids:
            return np.array([])

        p_fast_array = np.array([self._get_ema_from_shared_memory(tid, self.tracker)[0] for tid in task_ids])
        p_slow_array = np.array([self._get_ema_from_shared_memory(tid, self.tracker)[1] for tid in task_ids])

        # Apply reweighting if not at default (0.5)
        if abs(self.config.early_progress_amplification - 0.5) > 1e-6:
            # Apply reweighting element-wise to both fast and slow EMAs
            # Need to clip to [0, 1] range to ensure valid probability values
            p_fast_clipped = np.clip(p_fast_array, 0.0, 1.0)
            p_slow_clipped = np.clip(p_slow_array, 0.0, 1.0)

            # Vectorize the reweight function and apply
            reweighted_fast = np.vectorize(self._reweight)(p_fast_clipped)
            reweighted_slow = np.vectorize(self._reweight)(p_slow_clipped)

            # Calculate LP from reweighted signals
            return np.abs(reweighted_fast - reweighted_slow)
        else:
            # Default behavior: unweighted LP
            return np.abs(p_fast_array - p_slow_array)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function with clipping to prevent overflow."""
        x_clipped = np.clip(x, -500, 500)  # Clip to prevent overflow
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def _calculate_task_distribution(self) -> None:
        """Calculate task sampling distribution from learning progress."""
        if self.tracker is None:
            self._task_dist = None
            self._raw_lp_scores = None
            self._postzscored_lp_scores = None
            self._stale_dist = False
            return

        task_ids = self.tracker.get_all_tracked_tasks()
        if not task_ids:
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
        if self.config.performance_bonus_weight > 0 and self.tracker is not None:
            # Build p_true array from shared memory (using same task_ids from above)
            p_true_array = np.array([self._get_ema_from_shared_memory(tid, self.tracker)[2] for tid in task_ids])
            performance_bonus = p_true_array * self.config.performance_bonus_weight
            subprobs = subprobs + performance_bonus

        # Store raw LP scores (before z-score normalization)
        self._raw_lp_scores = subprobs.copy().astype(np.float32)

        # Apply temperature scaling or z-score normalization before sigmoid
        # Temperature controls how LP scores are transformed before sigmoid:
        # - temp > 0: Divide by temperature (low temp amplifies differences)
        # - temp = 0: Z-score normalize (standardize to mean=0, std=1) then amplify
        temperature = self.config.lp_score_temperature
        if temperature == 0:
            # Z-score normalization: center at mean and normalize by std
            # This makes sigmoid operate on standardized scores, preventing saturation
            mean = np.mean(subprobs)
            std = np.std(subprobs)
            if std > 1e-10:  # Avoid division by zero
                subprobs = (subprobs - mean) / std
                # Apply z-score amplification to restore selectivity
                # Higher values spread out the distribution more before sigmoid
                subprobs = subprobs * self.config.z_score_amplification
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

    def __init__(self, config: "LearningProgressConfig", tracker: Optional[TaskTracker] = None):
        """Initialize basic scorer.

        Args:
            config: Learning progress configuration
            tracker: Optional TaskTracker instance (for consistency with BidirectionalLPScorer)
        """
        super().__init__(config, tracker)

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
