from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from omegaconf import DictConfig

from mettagrid.curriculum.multi_task import MultiTaskCurriculum

from .random import RandomCurriculum

logger = logging.getLogger(__name__)


class LearningProgressCurriculum(MultiTaskCurriculum):
    """Curriculum that adaptively samples tasks to focus on low-reward scenarios."""

    def __init__(
        self,
        tasks: Dict[str, float],
        env_overrides: DictConfig,
        ema_alpha=0.001,
        p_theta=0.05,
        num_active_tasks=16,
        rand_task_rate=0.25,
        sample_threshold=10,
        memory=25,
    ):
        super().__init__(tasks, env_overrides)
        self._num_tasks = len(tasks)
        self._ema_alpha = ema_alpha
        self._p_theta = p_theta
        self._num_active_tasks = int(num_active_tasks)
        self._rand_task_rate = rand_task_rate
        self._sample_threshold = sample_threshold
        self._memory = int(memory)
        self._outcomes = {}
        for i in range(self._num_tasks):
            self._outcomes[i] = []
        self.ema_tsr = None
        self._p_fast = None
        self._p_slow = None
        self._p_true = None
        self._random_baseline = None
        self._task_success_rate = np.zeros(self._num_tasks)
        self._task_sampled_tracker = self._num_tasks * [0]
        self._mean_samples_per_eval = []
        self._num_nans = []

        # should we continue collecting
        #  or if we have enough data to update the learning progress
        self._collecting = True
        self._update_mask = np.ones(self._num_tasks).astype(bool)
        self._sample_levels = np.arange(self._num_tasks).astype(np.int32)
        self._counter = {i: 0 for i in self._sample_levels}

    def complete_task(self, id: str, score: float):
        self._outcomes[id].append(score)
        if id in self._sample_levels:
            self._counter[id] += 1
        super().complete_task(id, score)

    def add_stats(self, info):
        info['lp/num_active_tasks'] = len(self._sample_levels)
        info['lp/mean_sample_prob'] = np.mean(self._task_dist)
        info['lp/num_zeros_lp_dist'] = np.sum(self._task_dist == 0)
        info['lp/task_1_success_rate'] = self._task_success_rate[0]
        info[f'lp/task_{self._num_tasks // 2}_success_rate'] = self._task_success_rate[self._num_tasks // 2]
        info['lp/last_task_success_rate'] = self._task_success_rate[-1]
        info['lp/task_success_rate'] = np.mean(self._task_success_rate)
        info['lp/mean_evals_per_task'] = self._mean_samples_per_eval[-1]
        info['lp/num_nan_tasks'] = self._num_nans[-1]

    def _update(self):
        task_success_rates = np.array([np.mean(self._outcomes[i]) for i in range(self._num_tasks)])
        update_mask = self._update_mask

        if self._random_baseline is None:
            # Assume that any perfect success rate is actually 75% due to evaluation precision.
            # Prevents NaN probabilities and prevents task from being completely ignored.
            self._random_baseline = np.minimum(task_success_rates, 0.75)
            self.task_rates = task_success_rates

        # Update task scores
        normalized_task_success_rates = np.maximum(
            task_success_rates[update_mask] - self._random_baseline[update_mask],
            np.zeros(task_success_rates[update_mask].shape)) / (1.0 - self._random_baseline[update_mask])

        if self._p_fast is None:
            # Initial values
            self._p_fast = normalized_task_success_rates[update_mask]
            self._p_slow = normalized_task_success_rates[update_mask]
            self._p_true = task_success_rates[update_mask]
        else:
            # Exponential moving average
            self._p_fast[update_mask] = (normalized_task_success_rates * self.ema_alpha) + (self._p_fast[update_mask] * (1.0 - self.ema_alpha))
            self._p_slow[update_mask] = (self._p_fast[update_mask] * self.ema_alpha) + (self._p_slow[update_mask] * (1.0 - self.ema_alpha))
            self._p_true[update_mask] = (task_success_rates[update_mask] * self.ema_alpha) + (self._p_true[update_mask] * (1.0 - self.ema_alpha))

        self._task_rates[update_mask] = task_success_rates[update_mask]
        self._stale_dist = True
        self._task_dist = None

        return task_success_rates

    def _learning_progress(self, reweight: bool = True) -> float:
        """
        Compute the learning progress metric for the given task.
        """
        fast = self._reweight(self._p_fast) if reweight else self._p_fast
        slow = self._reweight(self._p_slow) if reweight else self._p_slow

        return abs(fast - slow)

    def _reweight(self, p: np.ndarray) -> float:
        """
        Reweight the given success rate using the reweighting function from the paper.
        """
        numerator = p * (1.0 - self._p_theta)
        denominator = p + self._p_theta * (1.0 - 2.0 * p)
        return numerator / denominator

    def _sigmoid(self, x: np.ndarray):
        """ Sigmoid function for reweighting the learning progress."""
        return 1 / (1 + np.exp(-x))

    def _sample_distribution(self):
        """ Return sampling distribution over the task space based on the learning progress."""
        task_dist = np.ones(self._num_tasks) / self._num_tasks

        learning_progress = self._learning_progress()
        pos_idxs = [i for i, lp in enumerate(learning_progress) if lp > 0 or self._p_true[i] > 0]
        any_progress = len(pos_idxs) > 0

        sub_probs = learning_progress[pos_idxs] if any_progress else learning_progress
        std = np.std(sub_probs)
        sub_probs = (sub_probs - np.mean(sub_probs)) / (std if std else 1)  # z-score
        sub_probs = self._sigmoid(sub_probs)  # sigmoid
        sub_probs = sub_probs / np.sum(sub_probs)  # normalize
        if any_progress:
            # If some tasks have nonzero progress, zero out the rest
            task_dist = np.zeros(len(learning_progress))
            task_dist[pos_idxs] = sub_probs
        else:
            # If all tasks have 0 progress, return uniform distribution
            task_dist = sub_probs

        self._task_dist = task_dist.astype(np.float32)
        self._stale_dist = False
        out_vec = [np.mean(self._outcomes[i]) for i in range(self._num_tasks)]
        self._num_nans.append(sum(np.isnan(out_vec)))
        self._task_success_rate = np.nan_to_num(out_vec)
        self._mean_samples_per_eval.append(np.mean([len(self._outcomes[i]) for i in range(self._num_tasks)]))
        for i in range(self._num_tasks):
            self._outcomes[i] = self._outcomes[i][-self._memory:]
        self._collecting = True
        return self._task_dist

    def _sample_tasks(self):
        sample_levels = []
        self._update_mask = np.zeros(self._num_tasks).astype(bool)
        for i in range(self._num_active_tasks):
            if np.random.rand() < self._rand_task_rate:
                level = np.random.choice(range(self._num_tasks))
            else:
                level = np.random.choice(range(self._num_tasks), p=self._task_dist)
            sample_levels.append(level)
            self._update_mask[level] = True
        self._sample_levels = np.array(sample_levels).astype(np.int32)
        self._counter = {i: 0 for i in self._sample_levels}
        return self._sample_levels

    def calculate_dist(self):
        if all([v < self._sample_threshold for k, v in self._counter.items()]) and self._random_baseline is not None:
            # collect more data on the current batch of tasks
            return self._task_dist, self._sample_levels
        self._task_success_rate = self._update()
        dist = self._sample_distribution()
        tasks = self._sample_tasks()
        return dist, tasks

    def reset_outcomes(self):
        self._prev_outcomes = self._outcomes
        self._outcomes = {}
        for i in range(self._num_tasks):
            self._outcomes[i] = []
