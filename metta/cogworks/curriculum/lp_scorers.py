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
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from .task_tracker import TaskTracker

if TYPE_CHECKING:
    from .learning_progress_algorithm import LearningProgressConfig

# Constants for bidirectional learning progress
DEFAULT_SUCCESS_RATE = 0.0


class LPScorer(ABC):
    """Abstract base class for learning progress scoring strategies."""

    def __init__(self, config: "LearningProgressConfig", tracker: TaskTracker):
        """Initialize scorer with configuration.

        Args:
            config: Learning progress configuration object
            tracker: TaskTracker instance for task performance data access
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


class BidirectionalLPScorer(LPScorer):
    """Bidirectional learning progress using fast/slow exponential moving averages.

    This scorer tracks recent task outcomes and computes learning progress
    by comparing fast and slow EMAs of performance. Tasks with positive
    learning progress (fast EMA > slow EMA) are prioritized.
    """

    def __init__(self, config: "LearningProgressConfig", tracker: TaskTracker):
        """Initialize bidirectional scorer.

        Args:
            config: Learning progress configuration
            tracker: TaskTracker instance for task performance data access
        """
        super().__init__(config, tracker)

        # Bidirectional learning progress tracking
        # All EMA state (p_fast, p_slow, p_true, random_baseline) is stored in shared memory (indices 13-16).
        # All LP scores stored in shared memory (index 4: lp_score).
        #
        # Single Source of Truth: Shared memory via TaskTracker
        # - EMAs read from shared memory on demand
        # - Completion counts read from shared memory on demand
        # - LP scores read from shared memory on demand
        #
        # Distribution Recalculation:
        # - Distribution recalculated when stale (after any task update)
        # - Recalculation includes z-score normalization + sigmoid across all tasks
        # - Final scores written back to shared memory atomically

        # Distribution staleness flag (only state we keep)
        self._stale_dist = True

    def _get_ema(self, task_id: int) -> tuple[float, float, float, float]:
        """Get EMA values for a task.

        Returns:
            Tuple of (p_fast, p_slow, p_true, random_baseline)
        """
        task_stats = self.tracker.get_task_stats(task_id)
        if task_stats is None:
            return 0.0, 0.0, 0.0, 0.0
        return (
            task_stats.get("p_fast", 0.0),
            task_stats.get("p_slow", 0.0),
            task_stats.get("p_true", 0.0),
            task_stats.get("random_baseline", 0.0),
        )

    def score_task(self, task_id: int, tracker: TaskTracker) -> float:
        """Calculate bidirectional learning progress score for a task.

        Reads from shared memory when distribution is fresh, recalculates when stale.
        This is the final sampling probability after all transformations.
        """
        # Check completion count from shared memory for sufficient data
        task_stats = tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < self.config.min_samples_for_lp:
            # Tasks without sufficient data get exploration bonus
            return self.config.exploration_bonus

        # If distribution is stale, recalculate for all tasks
        if self._stale_dist:
            self._calculate_task_distribution(tracker)

        # Read LP score from shared memory (written by _calculate_task_distribution)
        task_stats = tracker.get_task_stats(task_id)
        if task_stats:
            return task_stats.get("lp_score", self.config.exploration_bonus)
        else:
            return self.config.exploration_bonus

    def update_with_score(self, task_id: int, score: float) -> None:
        """Mark distribution as stale after task update.

        Stage 3: EMA updates now happen atomically in TaskTracker.
        This method marks the distribution as stale for recalculation.
        """
        # Mark distribution as stale - it will be recalculated on next score_task() call
        self._stale_dist = True

    def remove_task(self, task_id: int) -> None:
        """Remove task from scoring system."""
        # Mark distribution as stale - it will be recalculated on next score_task() call
        self._stale_dist = True

    def get_stats(self) -> Dict[str, float]:
        """Get detailed bidirectional learning progress statistics."""
        task_ids = self.tracker.get_all_tracked_tasks()
        if not task_ids:
            return {
                "mean_task_success_rate": 0.0,
                "mean_learning_progress": 0.0,
                "mean_lp_score": 0.0,
            }

        # Compute mean task success rate from shared memory success_rate_ema
        task_success_rates = []
        lp_scores = []
        for task_id in task_ids:
            task_stats = self.tracker.get_task_stats(task_id)
            if task_stats and task_stats["completion_count"] > 0:
                task_success_rates.append(task_stats["success_rate_ema"])
                lp_scores.append(task_stats.get("lp_score", 0.0))

        stats = {
            "mean_task_success_rate": float(np.mean(task_success_rates)) if task_success_rates else 0.0,
            "mean_lp_score": float(np.mean(lp_scores)) if lp_scores else 0.0,
        }

        # Calculate learning progress from shared memory
        # Pass task_ids to prevent race condition in multi-process environment
        learning_progress = self._learning_progress(task_ids)
        if len(learning_progress) > 0:
            # Count zeros in lp_scores list (convert to numpy for element-wise comparison)
            num_zeros = float(np.sum(np.array(lp_scores) == 0)) if lp_scores else 0.0
            stats.update(
                {
                    "mean_sample_prob": stats["mean_lp_score"],  # Approximate (after normalization)
                    "num_zeros_lp_dist": num_zeros,
                    "mean_learning_progress": float(np.mean(learning_progress)),
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
        """Serialize bidirectional scorer state.

        Note: All state (EMAs, LP scores) is stored in shared memory via the task tracker,
        so there's minimal state to serialize. We just mark distribution as stale on load.
        """
        return {
            "stale_dist": True,  # Always stale after load (force recalculation)
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Deserialize bidirectional scorer state.

        Note: All state (EMAs, LP scores) is stored in shared memory via the task tracker,
        so they're restored from there automatically. Just mark distribution as stale.
        """
        self._stale_dist = True  # Always recalculate distribution after load

    def invalidate_cache(self) -> None:
        """Invalidate distribution (mark as stale)."""
        self._stale_dist = True

    def _update_task_emas(self, task_id: int, score: float) -> None:
        """Update bidirectional EMAs for a single task with a new score.

        Args:
            task_id: ID of the task to update
            score: New performance score for the task
        """
        # Get current EMA values
        p_fast, p_slow, p_true, random_baseline = self._get_ema(task_id)

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

        # Write updated EMAs back to tracker
        self.tracker.update_bidirectional_emas(task_id, p_fast, p_slow, p_true, random_baseline)

        # Mark distribution as stale since EMAs changed
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

    def _learning_progress(self, task_ids: list[int] | None = None) -> np.ndarray:
        """Calculate raw learning progress with optional reweighting.

        Applies reweighting to fast and slow EMAs before computing absolute difference.
        This amplifies learning signal from unsolved tasks when early_progress_amplification
        is set to a low value (e.g., 0.05).

        Args:
            task_ids: Optional list of task IDs to calculate LP for. If None, fetches from tracker.
                     Providing task_ids prevents race conditions in multi-process environments.
        """
        # Build arrays from tracker
        if task_ids is None:
            task_ids = self.tracker.get_all_tracked_tasks()
        if not task_ids:
            return np.array([])

        p_fast_array = np.array([self._get_ema(tid)[0] for tid in task_ids])
        p_slow_array = np.array([self._get_ema(tid)[1] for tid in task_ids])

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

    def _calculate_task_distribution(self, tracker: TaskTracker) -> None:
        """Calculate task sampling distribution from learning progress and write to shared memory.

        This method reads EMAs from shared memory, calculates final LP scores,
        and writes them back atomically.
        """
        task_ids = tracker.get_all_tracked_tasks()
        if not task_ids:
            self._stale_dist = False
            return

        # Pass task_ids to prevent race condition in multi-process environment
        learning_progress = self._learning_progress(task_ids)
        if len(learning_progress) == 0:
            self._stale_dist = False
            return

        # Apply smoothing to all tasks (even those with negative/zero learning progress)
        # This ensures all tasks get non-zero probability
        subprobs = learning_progress + self.config.progress_smoothing

        # Add performance bonus if configured
        if self.config.performance_bonus_weight > 0:
            # Build p_true array (using same task_ids from above)
            p_true_array = np.array([self._get_ema(tid)[2] for tid in task_ids])
            performance_bonus = p_true_array * self.config.performance_bonus_weight
            subprobs = subprobs + performance_bonus

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

        subprobs = self._sigmoid(subprobs)

        # Normalize to sum to 1
        sum_probs = np.sum(subprobs)
        if sum_probs > 0:
            task_dist = subprobs / sum_probs
        else:
            task_dist = np.ones_like(subprobs) / len(subprobs)

        # Write LP scores back to shared memory atomically
        for i, task_id in enumerate(task_ids):
            lp_score = float(task_dist[i])
            tracker.update_lp_score(task_id, lp_score)

        self._stale_dist = False


class BasicLPScorer(LPScorer):
    """Basic learning progress using variance estimation from EMAs.

    This scorer computes learning progress as the variance in task performance,
    estimated from first and second moment EMAs. Higher variance indicates
    more learning opportunity.
    """

    def __init__(self, config: "LearningProgressConfig", tracker: TaskTracker):
        """Initialize basic scorer.

        Args:
            config: Learning progress configuration
            tracker: TaskTracker instance for task performance data access
        """
        super().__init__(config, tracker)

        # Stage 6: No local caching, only stale flag for consistency with BidirectionalLPScorer
        self._stale_dist = True

    def score_task(self, task_id: int, tracker: TaskTracker) -> float:
        """Calculate basic learning progress score using EMA variance.

        Stage 6: No local caching, always calculate from TaskTracker's shared memory.
        """
        task_stats = tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < 2:
            return self.config.exploration_bonus

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
            return self.config.exploration_bonus

        # Learning progress is approximated by variance in performance
        return std_dev

    def update_with_score(self, task_id: int, score: float) -> None:
        """Update basic EMA tracking for a task with new score.

        Stage 6: EMA updates now happen atomically in TaskTracker (Stage 3).
        This method only marks the distribution as stale for consistency.
        """
        # Mark distribution as stale (though basic mode doesn't use distributions)
        self._stale_dist = True

    def remove_task(self, task_id: int) -> None:
        """Remove task from scoring system."""
        # Mark distribution as stale
        self._stale_dist = True

    def get_stats(self) -> Dict[str, float]:
        """Get detailed basic learning progress statistics.

        Stage 6: Stats calculated from shared memory, no local caches.
        """
        # Basic mode stats are minimal since all data is in TaskTracker
        return {
            "mean_learning_progress": 0.0,  # Could compute from all tasks if needed
            "mean_lp_score": 0.0,
        }

    def get_state(self) -> Dict[str, Any]:
        """Serialize basic scorer state.

        Stage 6: Only stale flag is serialized, like BidirectionalLPScorer.
        """
        return {
            "stale_dist": self._stale_dist,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Deserialize basic scorer state.

        Stage 6: Only stale flag is restored.
        """
        self._stale_dist = state.get("stale_dist", True)

    def invalidate_cache(self) -> None:
        """Invalidate internal state (mark distribution as stale).

        Stage 6: No caches to clear, just mark as stale.
        """
        self._stale_dist = True
