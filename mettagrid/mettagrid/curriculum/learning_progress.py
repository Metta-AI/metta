from __future__ import annotations

from pdb import set_trace as T

import numpy as np
from collections import defaultdict
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import torch

import pufferlib
from gymnasium.spaces import Discrete

from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

from metta.util.config import config_from_path

from mettagrid.curriculum.curriculum import Task
from mettagrid.curriculum.multi_task import MultiTaskCurriculum
from mettagrid.curriculum.random import RandomCurriculum

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from mettagrid.mettagrid_env import MettaGridEnv


class LearningProgressCurriculum(RandomCurriculum):
    """Curriculum that adaptively samples tasks based on learning progress."""

    def __init__(self, tasks: Dict[str, float], env_overrides: DictConfig,
                 ema_timescale: float = 0.001, p_theta: float = 0.05,
                 num_active_tasks: int = 16, rand_task_rate: float = 0.25,
                 sample_threshold: int = 10, memory: int = 25):
        super().__init__(tasks, env_overrides)

        # Initialize learning progress tracker
        search_space_size = len(tasks)
        self.lp_tracker = BidirectionalLearningProgess(
            search_space=search_space_size,
            ema_timescale=ema_timescale,
            p_theta=p_theta,
            num_active_tasks=num_active_tasks,
            rand_task_rate=rand_task_rate,
            sample_threshold=sample_threshold,
            memory=memory
        )

        # Initialize task weights to uniform distribution
        self._task_weights = {task_id: 1.0 for task_id in tasks.keys()}
        self._normalize_weights()

        logger.info(f"LearningProgressCurriculum initialized with {search_space_size} tasks")

    def complete_task(self, id: str, score: float):
        """Complete a task and update learning progress tracking."""
        # Convert score to success rate (assuming score is between 0 and 1)
        success_rate = max(0.0, min(1.0, score))

        # Get task index for learning progress tracking
        task_idx = list(self._curriculums.keys()).index(id)

        # Collect data for learning progress
        self.lp_tracker.collect_data({f'tasks/{task_idx}': [success_rate]})

        # Update task distribution based on learning progress
        lp_weights, _ = self.lp_tracker.calculate_dist()
        if lp_weights is not None:
            # Update weights based on learning progress
            for i, task_id in enumerate(self._curriculums.keys()):
                if i < len(lp_weights):
                    self._task_weights[task_id] = lp_weights[i]

            # Normalize weights
            self._normalize_weights()

            logger.debug(f"Updated task weights based on learning progress: {self._task_weights}")

        super().complete_task(id, score)

    def _normalize_weights(self):
        """Normalize task weights to sum to 1."""
        total_weight = sum(self._task_weights.values())
        if total_weight > 0:
            self._task_weights = {k: v / total_weight for k, v in self._task_weights.items()}

    def get_stats(self) -> Dict[str, float]:
        """Get learning progress statistics for logging."""
        stats = {}
        if hasattr(self.lp_tracker, 'add_stats'):
            self.lp_tracker.add_stats(stats)
        return stats


class BidirectionalLearningProgess:
    def __init__(self, search_space, ema_timescale = 0.001, p_theta = 0.05, num_active_tasks = 16, rand_task_rate = 0.25,
                 sample_threshold = 10, memory = 25):
        if isinstance(search_space, int):
            search_space = Discrete(search_space)
        assert isinstance(search_space, Discrete), f"search_space must be a Discrete space or int, got {type(search_space)}"
        self.search_space = search_space
        self.num_tasks = max_num_levels = search_space.n
        self.ema_alpha = ema_timescale  # Fixed: use ema_timescale as ema_alpha
        self.p_theta = p_theta
        self.n = int(num_active_tasks)
        self.rand_task_rate = rand_task_rate
        self.sample_threshold = sample_threshold
        self.memory = int(memory)
        self.outcomes = {}
        for i in range(max_num_levels):
            self.outcomes[i] = []
        self.ema_tsr = None
        self._p_fast = None
        self._p_slow = None
        self._p_true = None
        self.random_baseline = None
        self.task_success_rate = np.zeros(max_num_levels)
        self.task_sampled_tracker = max_num_levels * [0]
        self.mean_samples_per_eval = []
        self.num_nans = []
        self.collecting = True
        self.update_mask = np.ones(max_num_levels).astype(bool)
        self.sample_levels = np.arange(max_num_levels).astype(np.int32)
        self.counter = {i: 0 for i in self.sample_levels}

    def add_stats(self, info):
        info['lp/num_active_tasks'] = len(self.sample_levels)
        info['lp/mean_sample_prob'] = np.mean(self.task_dist)
        info['lp/num_zeros_lp_dist'] = np.sum(self.task_dist == 0)
        info['lp/task_1_success_rate'] = self.task_success_rate[0]
        info[f'lp/task_{self.num_tasks // 2}_success_rate'] = self.task_success_rate[self.num_tasks // 2]
        info['lp/last_task_success_rate'] = self.task_success_rate[-1]
        info['lp/task_success_rate'] = np.mean(self.task_success_rate)
        info['lp/mean_evals_per_task'] = self.mean_samples_per_eval[-1]
        info['lp/num_nan_tasks'] = self.num_nans[-1]

    def _update(self):
        task_success_rates = np.array([np.mean(self.outcomes[i]) for i in range(self.num_tasks)])
        update_mask = self.update_mask

        if self.random_baseline is None:
            self.random_baseline = np.minimum(task_success_rates, 0.75)
            self.task_rates = task_success_rates

        normalized_task_success_rates = np.maximum(
            task_success_rates[update_mask] - self.random_baseline[update_mask],
            np.zeros(task_success_rates[update_mask].shape)) / (1.0 - self.random_baseline[update_mask])

        if self._p_fast is None:
            self._p_fast = normalized_task_success_rates[update_mask]
            self._p_slow = normalized_task_success_rates[update_mask]
            self._p_true = task_success_rates[update_mask]
        else:
            self._p_fast[update_mask] = (normalized_task_success_rates * self.ema_alpha) + (self._p_fast[update_mask] * (1.0 - self.ema_alpha))
            self._p_slow[update_mask] = (self._p_fast[update_mask] * self.ema_alpha) + (self._p_slow[update_mask] * (1.0 - self.ema_alpha))
            self._p_true[update_mask] = (task_success_rates[update_mask] * self.ema_alpha) + (self._p_true[update_mask] * (1.0 - self.ema_alpha))

        self.task_rates[update_mask] = task_success_rates[update_mask]
        self._stale_dist = True
        self.task_dist = None

        return task_success_rates

    def collect_data(self, infos):
        if not bool(infos):
            return

        for k, v in infos.items():
            if 'tasks' in k:
                task_id = int(k.split('/')[1])
                for res in v:
                    self.outcomes[task_id].append(res)
                    if task_id in self.sample_levels:
                        self.counter[task_id] += 1

    def continue_collecting(self):
        return self.collecting

    def _learning_progress(self, reweight: bool = True) -> float:
        fast = self._reweight(self._p_fast) if reweight else self._p_fast
        slow = self._reweight(self._p_slow) if reweight else self._p_slow
        return abs(fast - slow)

    def _reweight(self, p: np.ndarray) -> float:
        numerator = p * (1.0 - self.p_theta)
        denominator = p + self.p_theta * (1.0 - 2.0 * p)
        return numerator / denominator

    def _sigmoid(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def _sample_distribution(self):
        task_dist = np.ones(self.num_tasks) / self.num_tasks
        learning_progress = self._learning_progress()
        posidxs = [i for i, lp in enumerate(learning_progress) if lp > 0 or self._p_true[i] > 0]
        any_progress = len(posidxs) > 0
        subprobs = learning_progress[posidxs] if any_progress else learning_progress
        std = np.std(subprobs)
        subprobs = (subprobs - np.mean(subprobs)) / (std if std else 1)
        subprobs = self._sigmoid(subprobs)
        subprobs = subprobs / np.sum(subprobs)
        if any_progress:
            task_dist = np.zeros(len(learning_progress))
            task_dist[posidxs] = subprobs
        else:
            task_dist = subprobs
        self.task_dist = task_dist.astype(np.float32)
        self._stale_dist = False
        out_vec = [np.mean(self.outcomes[i]) for i in range(self.num_tasks)]
        self.num_nans.append(sum(np.isnan(out_vec)))
        self.task_success_rate = np.nan_to_num(out_vec)
        self.mean_samples_per_eval.append(np.mean([len(self.outcomes[i]) for i in range(self.num_tasks)]))
        for i in range(self.num_tasks):
            self.outcomes[i] = self.outcomes[i][-self.memory:]
        self.collecting = True
        return self.task_dist

    def _sample_tasks(self):
        sample_levels = []
        self.update_mask = np.zeros(self.num_tasks).astype(bool)
        for i in range(self.n):
            if np.random.rand() < self.rand_task_rate:
                level = np.random.choice(range(self.num_tasks))
            else:
                level = np.random.choice(range(self.num_tasks), p=self.task_dist)
            sample_levels.append(level)
            self.update_mask[level] = True
        self.sample_levels = np.array(sample_levels).astype(np.int32)
        self.counter = {i: 0 for i in self.sample_levels}
        return self.sample_levels

    def calculate_dist(self):
        if all([v < self.sample_threshold for k, v in self.counter.items()]) and self.random_baseline is not None:
            return self.task_dist, self.sample_levels
        self.task_success_rate = self._update()
        dist = self._sample_distribution()
        tasks = self._sample_tasks()
        return dist, tasks

    def reset_outcomes(self):
        self.prev_outcomes = self.outcomes
        self.outcomes = {}
        for i in range(self.num_tasks):
            self.outcomes[i] = []

