"""
Bidirectional Learning Progress scoring component.

Integrates bidirectional learning progress tracking with the existing curriculum system,
using fast and slow exponential moving averages to calculate learning progress scores.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from .task_tracker import TaskTracker

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SUCCESS_RATE = 0.0
DEFAULT_WEIGHT = 1.0
RANDOM_BASELINE_CAP = 0.75


class BidirectionalLearningProgressScorer:
    """Calculates bidirectional learning progress scores using fast and slow EMAs."""

    def __init__(
        self,
        ema_timescale: float = 0.001,
        exploration_bonus: float = 0.1,
        progress_smoothing: float = 0.05,
        num_active_tasks: int = 16,
        rand_task_rate: float = 0.25,
        sample_threshold: int = 10,
        memory: int = 25,
    ):
        self.ema_timescale = ema_timescale
        self.exploration_bonus = exploration_bonus
        self.progress_smoothing = progress_smoothing
        self.num_active_tasks = num_active_tasks
        self.rand_task_rate = rand_task_rate
        self.sample_threshold = sample_threshold
        self.memory = memory

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

    def update_task_ema(self, task_id: int, score: float) -> None:
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

    def get_learning_progress_score(self, task_id: int, task_tracker: TaskTracker) -> float:
        """Calculate bidirectional learning progress score for a task."""
        # Return cached score if valid
        if task_id in self._cache_valid_tasks and task_id in self._score_cache:
            return self._score_cache[task_id]

        task_stats = task_tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < 2:
            # New tasks get exploration bonus
            score = self.exploration_bonus
        elif task_id not in self._outcomes or len(self._outcomes[task_id]) < 2:
            # Tasks without sufficient data get exploration bonus
            score = self.exploration_bonus
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
                if task_idx < len(self._task_dist):
                    # Use the bidirectional learning progress as score
                    score = float(self._task_dist[task_idx])
                else:
                    score = self.exploration_bonus
            else:
                score = self.exploration_bonus

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
        """Recommend which task to evict based on bidirectional learning progress."""
        if not task_ids:
            return None

        scores = self.score_tasks(task_ids, task_tracker)

        # Find task with minimum learning progress
        min_task_id = min(task_ids, key=lambda tid: scores.get(tid, 0.0))
        return min_task_id

    def remove_task(self, task_id: int) -> None:
        """Remove task from tracking and clear its cache."""
        self._outcomes.pop(task_id, None)
        self._counter.pop(task_id, None)
        self._score_cache.pop(task_id, None)
        self._cache_valid_tasks.discard(task_id)
        self._stale_dist = True

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
            if len(self._p_fast) != num_tasks:
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
            self._p_fast[self._update_mask] = normalized_task_success_rates * self.ema_timescale + self._p_fast[
                self._update_mask
            ] * (1.0 - self.ema_timescale)
            self._p_slow[self._update_mask] = self._p_fast[self._update_mask] * self.ema_timescale + self._p_slow[
                self._update_mask
            ] * (1.0 - self.ema_timescale)
            self._p_true[self._update_mask] = task_success_rates[self._update_mask] * self.ema_timescale + self._p_true[
                self._update_mask
            ] * (1.0 - self.ema_timescale)

        self._task_success_rate = task_success_rates
        self._stale_dist = True

    def _learning_progress(self, reweight: bool = True) -> np.ndarray:
        """Calculate learning progress as the difference between fast and slow moving averages."""
        if self._p_fast is None or self._p_slow is None:
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

    def get_stats(self) -> Dict[str, float]:
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
