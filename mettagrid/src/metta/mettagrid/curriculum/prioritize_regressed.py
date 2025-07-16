from __future__ import annotations

import logging
from typing import Dict

from omegaconf import DictConfig

from metta.mettagrid.curriculum.random import RandomCurriculum

logger = logging.getLogger(__name__)


class PrioritizeRegressedCurriculum(RandomCurriculum):
    """Curriculum that prioritizes tasks where current performance has regressed relative to peak performance.

    This curriculum tracks both the maximum reward achieved and the moving average of rewards for each task.
    Tasks with high max/average ratios get higher weight, meaning tasks where we've seen good performance
    but are currently performing poorly get prioritized.
    """

    def __init__(
        self, tasks: Dict[str, float], env_overrides: DictConfig | None = None, moving_avg_decay_rate: float = 0.01
    ):
        super().__init__(tasks, env_overrides)
        self._reward_averages = {task_id: 0.0 for task_id in tasks.keys()}
        self._reward_maxes = {task_id: 0.0 for task_id in tasks.keys()}
        self._moving_avg_decay_rate = moving_avg_decay_rate  # Smoothing factor for moving average

    def complete_task(self, id: str, score: float):
        # Update moving average for the completed task
        old_average = self._reward_averages[id]
        self._reward_averages[id] = (1 - self._moving_avg_decay_rate) * self._reward_averages[
            id
        ] + self._moving_avg_decay_rate * score
        logger.debug(
            f"Updated task {id} "
            + f"reward mean({old_average:.3f} -> {self._reward_averages[id]:.3f}), "
            + f"max({self._reward_maxes[id]:.3f})"
        )
        self._reward_maxes[id] = max(self._reward_maxes[id], score)
        self._task_weights = {
            t: 1e-6 + self._reward_maxes[t] / (self._reward_averages[t] + 1e-6) for t in self._curricula.keys()
        }
        super().complete_task(id, score)
