from __future__ import annotations

import logging
from typing import Dict

from omegaconf import DictConfig

from metta.mettagrid.curriculum.random import RandomCurriculum

logger = logging.getLogger(__name__)


class LowRewardCurriculum(RandomCurriculum):
    """Curriculum that adaptively samples tasks to focus on low-reward scenarios."""

    def __init__(self, tasks: Dict[str, float], env_overrides: DictConfig):
        super().__init__(tasks, env_overrides)
        self._reward_averages = {task_id: 0.0 for task_id in tasks.keys()}
        self._reward_maxes = {task_id: 0.0 for task_id in tasks.keys()}
        self._alpha = 0.01  # Smoothing factor for moving average

    def complete_task(self, id: str, score: float):
        # Update moving average for the completed task
        old_average = self._reward_averages[id]
        self._reward_averages[id] = (1 - self._alpha) * self._reward_averages[id] + self._alpha * score
        logger.debug(
            f"Updated task {id} "
            + f"reward mean({old_average:.3f} -> {self._reward_averages[id]:.3f}), "
            + f"max({self._reward_maxes[id]:.3f})"
        )
        self._reward_maxes[id] = max(self._reward_maxes[id], score)
        self._task_weights = {
            t: 1e-6 + self._reward_maxes[t] / (self._reward_averages[t] + 1e-6) for t in self._curriculums.keys()
        }
        super().complete_task(id, score)
