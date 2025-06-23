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
        # Log sampling probabilities for each task to WandB
        if wandb.run is not None:
            total_weight = sum(self._task_weights.values())
            if total_weight > 0:
                task_probs = {k: v / total_weight for k, v in self._task_weights.items()}
                wandb.run.log({"curriculum/task_probs": task_probs}, commit=False)
        return task
