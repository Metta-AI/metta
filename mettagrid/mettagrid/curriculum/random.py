from __future__ import annotations

import logging
import random

from mettagrid.curriculum.curriculum import Task
from mettagrid.curriculum.multi_task import MultiTaskCurriculum

logger = logging.getLogger(__name__)


class RandomCurriculum(MultiTaskCurriculum):
    """Curriculum that samples from multiple environment types with fixed weights."""

    def get_task(self) -> Task:
        task_id = random.choices(list(self._curriculums.keys()), weights=list(self._task_weights.values()))[0]
        task = self._curriculums[task_id].get_task()
        task.add_parent(self, task_id)
        logger.debug(f"Task selected: {task.name()}")
        return task

    def get_task_probs(self):
        total = sum(self._task_weights.values())
        if total == 0:
            return {k: 0.0 for k in self._task_weights}
        return {k: v / total for k, v in self._task_weights.items()}
