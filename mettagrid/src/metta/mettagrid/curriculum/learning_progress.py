from __future__ import annotations

import collections
import logging

import numpy as np
from omegaconf import DictConfig

from metta.common.util.typed_config import BaseModelWithForbidExtra
from metta.mettagrid.curriculum.random import RandomCurriculum

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SUCCESS_RATE = 0.0
DEFAULT_WEIGHT = 1.0
RANDOM_BASELINE_CAP = 0.75


class LearningProgressCurriculumConfig(BaseModelWithForbidExtra):
    ema_timescale: float = 0.001
    progress_smoothing: float = 0.05
    num_active_tasks: int = 16
    rand_task_rate: float = 0.25
    sample_threshold: int = 10
    memory: int = 25


class LearningProgressCurriculum(RandomCurriculum):
    """Curriculum that adaptively samples tasks based on learning progress."""

    def __init__(
        self,
        tasks: dict[str, float] | DictConfig,
        env_overrides: DictConfig | None = None,
        **kwargs,
    ):
        super().__init__(tasks, env_overrides)

        config = LearningProgressCurriculumConfig(**kwargs)

        # Initialize learning progress tracker
        num_tasks = len(tasks)
        self._lp_tracker = BidirectionalLearningProgress(
            num_tasks=num_tasks,
            config=config,
        )

        logger.info(f"LearningProgressCurriculum initialized with {num_tasks} tasks")

    def complete_task(self, id: str, score: float):
        """Complete a task and update learning progress tracking."""
        # Convert score to success rate (assuming score is between 0 and 1)
        success_rate = max(0.0, min(1.0, score))

        # Get task index for learning progress tracking
        task_idx = list(self._curricula.keys()).index(id)

        # Collect data for learning progress
        self._lp_tracker.complete_task(task_idx, success_rate)

        # Update task weights based on learning progress
        lp_weights, _ = self._lp_tracker.calculate_dist()

        # Update weights based on learning progress
        for i, task_id in enumerate(self._curricula.keys()):
            if i < len(lp_weights):
                self._task_weights[task_id] = lp_weights[i]

        # Normalize weights
        total_weight = sum(self._task_weights.values())
        if total_weight > 0:
            self._task_weights = {k: v / total_weight for k, v in self._task_weights.items()}

        super().complete_task(id, score)

    def get_curriculum_stats(self) -> dict[str, float]:
        """Get learning progress statistics for logging."""
        return self._lp_tracker.get_stats()


class BidirectionalLearningProgress:
    """Tracks bidirectional learning progress using fast and slow exponential moving averages."""

    def __init__(
        self,
        *,
        num_tasks: int,
        config: LearningProgressCurriculumConfig,
    ) -> None:
        self._num_tasks = num_tasks
        self.config = config

        self._outcomes = [collections.deque[float](maxlen=self.config.memory) for _ in range(num_tasks)]
        self._p_fast = None
        self._p_slow = None
        self._p_true = None
        self._random_baseline = None
        self._task_success_rate = np.zeros(num_tasks)
        self._mean_samples_per_eval = []
        self._update_mask = np.ones(num_tasks).astype(bool)
        self._sample_levels = np.arange(num_tasks).astype(np.int32)
        self._counter = {i: 0 for i in self._sample_levels}
        self._task_dist = None

    def get_stats(self) -> dict[str, float]:
        """Return learning progress statistics for logging."""
        stats: dict[str, float] = {}
        stats["lp/num_active_tasks"] = len(self._sample_levels)
        stats["lp/mean_sample_prob"] = np.mean(self._task_dist or [])
        stats["lp/num_zeros_lp_dist"] = 0 if self._task_dist is None else np.sum(self._task_dist == 0)
        stats["lp/task_1_success_rate"] = self._task_success_rate[0]
        stats[f"lp/task_{self._num_tasks // 2}_success_rate"] = self._task_success_rate[self._num_tasks // 2]
        stats["lp/last_task_success_rate"] = self._task_success_rate[-1]
        stats["lp/task_success_rate"] = np.mean(self._task_success_rate)
        stats["lp/mean_evals_per_task"] = self._mean_samples_per_eval[-1]
        return stats

    def _update(self):
        """Update learning progress tracking with current task success rates."""
        task_success_rates = np.array(
            [
                np.mean(task_outcomes) if len(task_outcomes) > 0 else DEFAULT_SUCCESS_RATE
                for task_outcomes in self._outcomes
            ]
        )

        if self._random_baseline is None:
            self._random_baseline = np.minimum(task_success_rates, RANDOM_BASELINE_CAP)

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

        if self._p_fast is None or self._p_slow is None or self._p_true is None:
            self._p_fast = normalized_task_success_rates[self._update_mask]
            self._p_slow = normalized_task_success_rates[self._update_mask]
            self._p_true = task_success_rates[self._update_mask]
        else:
            self._p_fast[self._update_mask] = (normalized_task_success_rates * self.config.ema_timescale) + (
                self._p_fast[self._update_mask] * (1.0 - self.config.ema_timescale)
            )
            self._p_slow[self._update_mask] = (self._p_fast[self._update_mask] * self.config.ema_timescale) + (
                self._p_slow[self._update_mask] * (1.0 - self.config.ema_timescale)
            )
            self._p_true[self._update_mask] = (task_success_rates[self._update_mask] * self.config.ema_timescale) + (
                self._p_true[self._update_mask] * (1.0 - self.config.ema_timescale)
            )

        self._task_dist = None

        return task_success_rates

    def complete_task(self, task_id: int, res: float):
        """Collect task outcome data for learning progress tracking."""
        self._outcomes[task_id].append(res)
        if task_id in self._sample_levels:
            self._counter[task_id] += 1

    def _learning_progress(self, reweight: bool = True) -> np.ndarray:
        """Calculate learning progress as the difference between fast and slow moving averages."""
        fast = self._reweight(self._p_fast) if reweight else self._p_fast
        slow = self._reweight(self._p_slow) if reweight else self._p_slow
        return abs(fast - slow)

    def _reweight(self, probs: np.ndarray) -> np.ndarray:
        """Apply progress smoothing reweighting to probability values."""
        numerator = probs * (1.0 - self.config.progress_smoothing)
        denominator = probs + self.config.progress_smoothing * (1.0 - 2.0 * probs)

        # Handle division by zero
        denominator = np.where(denominator <= 0, 1.0, denominator)
        result = numerator / denominator

        return result

    def _sigmoid(self, x: np.ndarray):
        """Apply sigmoid function to array values."""
        return 1 / (1 + np.exp(-x))

    def _sample_distribution(self):
        task_dist = np.ones(self._num_tasks) / self._num_tasks
        learning_progress = self._learning_progress()

        posidxs = [i for i, lp in enumerate(learning_progress) if lp > 0 or self._p_true[i] > 0]
        any_progress = len(posidxs) > 0
        subprobs = learning_progress[posidxs] if any_progress else learning_progress

        std = np.std(subprobs)
        if std > 0:
            subprobs = (subprobs - np.mean(subprobs)) / std
        else:
            # If all values are the same, keep them as is
            subprobs = subprobs - np.mean(subprobs)

        subprobs = self._sigmoid(subprobs)

        # Normalize to sum to 1, handling zero sum case
        sum_probs = np.sum(subprobs)
        if sum_probs > 0:
            subprobs = subprobs / sum_probs
        else:
            # If all probabilities are zero, use uniform distribution
            subprobs = np.ones_like(subprobs) / len(subprobs)

        if any_progress:
            task_dist = np.zeros(len(learning_progress))
            task_dist[posidxs] = subprobs
        else:
            task_dist = subprobs

        self._task_dist = task_dist.astype(np.float32)

        out_vec = [
            np.mean(task_outcomes) if len(task_outcomes) > 0 else DEFAULT_SUCCESS_RATE
            for task_outcomes in self._outcomes
        ]
        out_vec = [DEFAULT_SUCCESS_RATE if np.isnan(x) else x for x in out_vec]  # Handle NaN in outcomes
        self._task_success_rate = np.array(out_vec)
        self._mean_samples_per_eval.append(np.mean([len(t) for t in self._outcomes]))

        return self._task_dist

    def _sample_tasks(self):
        """Sample active tasks based on current task distribution."""
        sample_levels = []
        self._update_mask = np.zeros(self._num_tasks).astype(bool)

        # Ensure task_dist is valid
        if self._task_dist is None or len(self._task_dist) == 0:
            # Use uniform distribution if task_dist is not available
            task_dist = np.ones(self._num_tasks) / self._num_tasks
        else:
            task_dist = self._task_dist.copy()

        # Ensure task_dist sums to 1
        sum_dist = np.sum(task_dist)
        if sum_dist <= 0:
            task_dist = np.ones(self._num_tasks) / self._num_tasks
        else:
            task_dist = task_dist / sum_dist

        for _i in range(self.config.num_active_tasks):
            if np.random.rand() < self.config.rand_task_rate:
                level = np.random.choice(range(self._num_tasks))
            else:
                try:
                    level = np.random.choice(range(self._num_tasks), p=task_dist)
                except ValueError as e:
                    logger.warning(f"Error in np.random.choice: {e}, using uniform distribution")
                    level = np.random.choice(range(self._num_tasks))
            sample_levels.append(level)
            self._update_mask[level] = True
        self._sample_levels = np.array(sample_levels).astype(np.int32)
        self._counter = {i: 0 for i in self._sample_levels}
        return self._sample_levels

    def calculate_dist(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate task distribution and sample levels based on learning progress."""
        if (
            all([v < self.config.sample_threshold for k, v in self._counter.items()])
            and self._random_baseline is not None
        ):
            # Ensure we have valid task_dist and sample_levels
            if self._task_dist is None or len(self._task_dist) == 0:
                self._task_dist = np.ones(self._num_tasks) / self._num_tasks
            if len(self._sample_levels) == 0:
                self._sample_levels = np.arange(self._num_tasks).astype(np.int32)
            return self._task_dist, self._sample_levels

        self._task_success_rate = self._update()
        task_dist = self._sample_distribution()
        tasks = self._sample_tasks()

        return task_dist, tasks
