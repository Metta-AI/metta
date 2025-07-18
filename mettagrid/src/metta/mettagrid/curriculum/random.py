from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.core import Curriculum, Task
from metta.mettagrid.curriculum.multi_task import MultiTaskCurriculum
from metta.mettagrid.curriculum.util import curriculum_from_config_path

logger = logging.getLogger(__name__)


class RandomCurriculum(MultiTaskCurriculum):
    """Curriculum that samples from multiple environment types with fixed weights."""

    def __init__(self, tasks: Dict[str, float], env_overrides: DictConfig | None = None):
        self.env_overrides = env_overrides or OmegaConf.create({})
        curricula = {t: self._curriculum_from_id(t) for t in tasks.keys()}
        self._task_weights = tasks
        super().__init__(curricula)

    def get_task(self, random_task_number: int = 1) -> Task:
        rng = np.random.RandomState(random_task_number)
        task_ids = list(self._curricula.keys())
        weights = list(self._task_weights.values())
        task_id = rng.choice(task_ids, p=np.array(weights) / np.sum(weights))
        task = self._curricula[task_id].get_task()
        task.add_parent(self, task_id)
        logger.debug(f"Task selected: {task.name()}")
        return task

    def _curriculum_from_id(self, cfg_path: str) -> Curriculum:
        return curriculum_from_config_path(cfg_path, self.env_overrides)
