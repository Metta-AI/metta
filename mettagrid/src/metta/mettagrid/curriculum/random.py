from __future__ import annotations

import logging
import random

from mmetta.mettagrid.curriculum.multi_task import MultiTaskCurriculum

from metta.mettagrid.curriculum.core import Task

logger = logging.getLogger(__name__)


class RandomCurriculum(MultiTaskCurriculum):
    """Curriculum that samples from multiple environment types with fixed weights."""

    def get_task(self) -> Task:
        task_id = random.choices(list(self._curriculums.keys()), weights=list(self._task_weights.values()))[0]
        task = self._curriculums[task_id].get_task()
        task.add_parent(self, task_id)
        logger.debug(f"Task selected: {task.name()}")
        return task
