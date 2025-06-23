from __future__ import annotations

import logging
import random
import wandb

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
        # Minimal WandB logging for curriculum task selection
        if wandb.run is not None:
            wandb.run.log({"curriculum/task_id": task_id}, commit=False)
            wandb.run.log({"curriculum/task_weights": dict(self._task_weights)}, commit=False)
        return task
