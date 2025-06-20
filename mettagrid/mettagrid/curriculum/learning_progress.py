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

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from mettagrid.mettagrid_env import MettaGridEnv


class LearningProgressCurriculum(MultiTaskCurriculum):
    """Curriculum that adaptively samples tasks based on learning progress."""

    def __init__(self, tasks: Dict[str, float], env_overrides: DictConfig,
                 ema_timescale: float = 0.001, progress_smoothing: float = 0.05,
                 num_active_tasks: int = 16, rand_task_rate: float = 0.25,
                 sample_threshold: int = 10, memory: int = 25):
        super().__init__(tasks, env_overrides)

        # Initialize learning progress tracker
        search_space_size = len(tasks)
        self.lp_tracker = BidirectionalLearningProgess(
            search_space=search_space_size,
            ema_timescale=ema_timescale,
            progress_smoothing=progress_smoothing,
            num_active_tasks=num_active_tasks,
            rand_task_rate=rand_task_rate,
            sample_threshold=sample_threshold,
            memory=memory
        )

        # Initialize task weights to uniform distribution
        self._task_weights = {task_id: 1.0 for task_id in tasks.keys()}
        self._normalize_weights()

        logger.info(f"LearningProgressCurriculum initialized with {search_space_size} tasks")

    def get_task(self) -> Task:
        """Get a task based on learning progress weights."""
        # Get current learning progress weights
        lp_weights, _ = self.lp_tracker.calculate_dist()

        # Debug logging
        logger.debug(f"LearningProgressCurriculum get_task called")
        logger.debug(f"  lp_weights: {lp_weights}")

        if lp_weights is not None and len(lp_weights) > 0:
            # Update weights based on learning progress
            for i, task_id in enumerate(self._curriculums.keys()):
                if i < len(lp_weights):
                    self._task_weights[task_id] = lp_weights[i]

            # Normalize weights
            self._normalize_weights()
        else:
            # If no learning progress data yet, use uniform weights
            logger.debug("No learning progress data yet, using uniform weights")
            num_tasks = len(self._curriculums)
            uniform_weight = 1.0 / num_tasks
            self._task_weights = {task_id: uniform_weight for task_id in self._curriculums.keys()}

        # Sample task based on current weights
        task_ids = list(self._curriculums.keys())

        # Safety check: ensure we have valid task IDs
        if not task_ids:
            raise ValueError("No tasks available in curriculum")

        weights = [self._task_weights[task_id] for task_id in task_ids]

        # Handle NaN values in weights
        weights = [0.0 if np.isnan(w) else w for w in weights]

        # Ensure weights sum to a positive value
        total_weight = sum(weights)
        if total_weight <= 0:
            # If all weights are zero or negative, use uniform distribution
            logger.warning("All weights are zero or negative, using uniform distribution")
            weights = [1.0] * len(weights)
            total_weight = len(weights)

        # Normalize weights to sum to 1
        weights = [w / total_weight for w in weights]

        # Debug logging
        logger.debug(f"LearningProgressCurriculum task selection:")
        logger.debug(f"  Available tasks: {task_ids}")
        logger.debug(f"  Task weights: {weights}")
        logger.debug(f"  Total weight: {total_weight}")

        try:
            task_id = random.choices(task_ids, weights=weights)[0]
        except (ValueError, IndexError) as e:
            logger.error(f"Error in random.choices: {e}, falling back to random task")
            task_id = random.choices(task_ids)[0]

        # Additional debug check
        if task_id is None:
            logger.error("Selected task_id is None! Falling back to first task.")
            task_id = task_ids[0] if task_ids else None

        if task_id is None:
            raise ValueError("No valid task ID available in curriculum")

        logger.debug(f"  Selected task: {task_id}")

        # Get the actual task from the curriculum
        if task_id not in self._curriculums:
            logger.error(f"Task ID {task_id} not found in curriculums: {list(self._curriculums.keys())}")
            # Fall back to first available task
            task_id = list(self._curriculums.keys())[0]

        task = self._curriculums[task_id].get_task()

        # Safety check: ensure task is not None
        if task is None:
            raise ValueError(f"Curriculum {task_id} returned None task")

        task.add_parent(self, task_id)
        logger.debug(f"Task selected: {task.name()}")
        return task

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

        logger.debug(f"Updated learning progress for task {id} with score {score}")

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
    def __init__(self, search_space, ema_timescale = 0.001, progress_smoothing = 0.05, num_active_tasks = 16, rand_task_rate = 0.25,
                 sample_threshold = 10, memory = 25):
        if isinstance(search_space, int):
            search_space = Discrete(search_space)
        assert isinstance(search_space, Discrete), f"search_space must be a Discrete space or int, got {type(search_space)}"
        self.search_space = search_space
        self.num_tasks = max_num_levels = search_space.n
        self.ema_alpha = ema_timescale  # Fixed: use ema_timescale as ema_alpha
        self.progress_smoothing = progress_smoothing
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
        # Handle NaN values in task success rates
        task_success_rates = np.nan_to_num(task_success_rates, nan=0.0)
        update_mask = self.update_mask

        if self.random_baseline is None:
            self.random_baseline = np.minimum(task_success_rates, 0.75)
            self.task_rates = task_success_rates

        # Handle division by zero and NaN in normalization
        denominator = (1.0 - self.random_baseline[update_mask])
        denominator = np.where(denominator <= 0, 1.0, denominator)  # Avoid division by zero

        normalized_task_success_rates = np.maximum(
            task_success_rates[update_mask] - self.random_baseline[update_mask],
            np.zeros(task_success_rates[update_mask].shape)) / denominator

        # Handle NaN values in normalized rates
        normalized_task_success_rates = np.nan_to_num(normalized_task_success_rates, nan=0.0)

        if self._p_fast is None:
            self._p_fast = normalized_task_success_rates[update_mask]
            self._p_slow = normalized_task_success_rates[update_mask]
            self._p_true = task_success_rates[update_mask]
        else:
            self._p_fast[update_mask] = (normalized_task_success_rates * self.ema_alpha) + (self._p_fast[update_mask] * (1.0 - self.ema_alpha))
            self._p_slow[update_mask] = (self._p_fast[update_mask] * self.ema_alpha) + (self._p_slow[update_mask] * (1.0 - self.ema_alpha))
            self._p_true[update_mask] = (task_success_rates[update_mask] * self.ema_alpha) + (self._p_true[update_mask] * (1.0 - self.ema_alpha))

        # Handle NaN values in EMA updates
        self._p_fast = np.nan_to_num(self._p_fast, nan=0.0)
        self._p_slow = np.nan_to_num(self._p_slow, nan=0.0)
        self._p_true = np.nan_to_num(self._p_true, nan=0.0)

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
        # Handle NaN values in input
        p = np.nan_to_num(p, nan=0.0)

        numerator = p * (1.0 - self.progress_smoothing)
        denominator = p + self.progress_smoothing * (1.0 - 2.0 * p)

        # Handle division by zero
        denominator = np.where(denominator <= 0, 1.0, denominator)
        result = numerator / denominator

        # Handle NaN values in result
        result = np.nan_to_num(result, nan=0.0)
        return result

    def _sigmoid(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def _sample_distribution(self):
        task_dist = np.ones(self.num_tasks) / self.num_tasks
        learning_progress = self._learning_progress()

        # Handle NaN values in learning progress
        learning_progress = np.nan_to_num(learning_progress, nan=0.0)

        posidxs = [i for i, lp in enumerate(learning_progress) if lp > 0 or self._p_true[i] > 0]
        any_progress = len(posidxs) > 0
        subprobs = learning_progress[posidxs] if any_progress else learning_progress

        # Handle NaN values in subprobs
        subprobs = np.nan_to_num(subprobs, nan=0.0)

        std = np.std(subprobs)
        if std > 0:
            subprobs = (subprobs - np.mean(subprobs)) / std
        else:
            # If all values are the same, keep them as is
            subprobs = subprobs - np.mean(subprobs)

        # Handle NaN values after normalization
        subprobs = np.nan_to_num(subprobs, nan=0.0)

        subprobs = self._sigmoid(subprobs)

        # Handle NaN values after sigmoid
        subprobs = np.nan_to_num(subprobs, nan=0.0)

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

        # Final NaN check and normalization
        task_dist = np.nan_to_num(task_dist, nan=1.0/len(task_dist))
        sum_dist = np.sum(task_dist)
        if sum_dist > 0:
            task_dist = task_dist / sum_dist
        else:
            task_dist = np.ones(len(task_dist)) / len(task_dist)

        self.task_dist = task_dist.astype(np.float32)
        self._stale_dist = False

        out_vec = [np.mean(self.outcomes[i]) for i in range(self.num_tasks)]
        out_vec = [0.0 if np.isnan(x) else x for x in out_vec]  # Handle NaN in outcomes
        self.num_nans.append(sum(np.isnan(out_vec)))
        self.task_success_rate = np.array(out_vec)
        self.mean_samples_per_eval.append(np.mean([len(self.outcomes[i]) for i in range(self.num_tasks)]))

        for i in range(self.num_tasks):
            self.outcomes[i] = self.outcomes[i][-self.memory:]
        self.collecting = True
        return self.task_dist

    def _sample_tasks(self):
        sample_levels = []
        self.update_mask = np.zeros(self.num_tasks).astype(bool)

        # Ensure task_dist is valid
        if self.task_dist is None or len(self.task_dist) == 0:
            # Use uniform distribution if task_dist is not available
            task_dist = np.ones(self.num_tasks) / self.num_tasks
        else:
            task_dist = self.task_dist.copy()

        # Handle NaN values in task_dist
        task_dist = np.nan_to_num(task_dist, nan=1.0/len(task_dist))

        # Ensure task_dist sums to 1
        sum_dist = np.sum(task_dist)
        if sum_dist <= 0:
            task_dist = np.ones(self.num_tasks) / self.num_tasks
        else:
            task_dist = task_dist / sum_dist

        for i in range(self.n):
            if np.random.rand() < self.rand_task_rate:
                level = np.random.choice(range(self.num_tasks))
            else:
                try:
                    level = np.random.choice(range(self.num_tasks), p=task_dist)
                except ValueError as e:
                    logger.warning(f"Error in np.random.choice: {e}, using uniform distribution")
                    level = np.random.choice(range(self.num_tasks))
            sample_levels.append(level)
            self.update_mask[level] = True
        self.sample_levels = np.array(sample_levels).astype(np.int32)
        self.counter = {i: 0 for i in self.sample_levels}
        return self.sample_levels

    def calculate_dist(self):
        if all([v < self.sample_threshold for k, v in self.counter.items()]) and self.random_baseline is not None:
            logger.debug(f"Using existing task_dist: {self.task_dist}")
            # Ensure we have valid task_dist and sample_levels
            if self.task_dist is None or len(self.task_dist) == 0:
                self.task_dist = np.ones(self.num_tasks) / self.num_tasks
            if self.sample_levels is None or len(self.sample_levels) == 0:
                self.sample_levels = np.arange(self.num_tasks).astype(np.int32)
            return self.task_dist, self.sample_levels

        self.task_success_rate = self._update()
        dist = self._sample_distribution()
        tasks = self._sample_tasks()

        # Ensure we have valid return values
        if dist is None or len(dist) == 0:
            dist = np.ones(self.num_tasks) / self.num_tasks
        if tasks is None or len(tasks) == 0:
            tasks = np.arange(self.num_tasks).astype(np.int32)

        logger.debug(f"Calculated new task_dist: {dist}")
        logger.debug(f"Sample levels: {tasks}")
        return dist, tasks

    def reset_outcomes(self):
        self.prev_outcomes = self.outcomes
        self.outcomes = {}
        for i in range(self.num_tasks):
            self.outcomes[i] = []

