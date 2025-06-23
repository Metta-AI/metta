from __future__ import annotations

import logging
import random
from typing import Dict

from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import Curriculum, Task
from metta.mettagrid.curriculum.multi_task import MultiTaskCurriculum
from metta.mettagrid.curriculum.util import curriculum_from_config_path

logger = logging.getLogger(__name__)


class RandomCurriculum(MultiTaskCurriculum):
    """Curriculum that samples from multiple environment types with fixed weights."""

    def __init__(self, curricula_cfgs: Dict[str, float], env_overrides: DictConfig):
        self.env_overrides = env_overrides
        curricula = {t: self._curriculum_from_id(t) for t in curricula_cfgs.keys()}
        self._task_weights = curricula_cfgs
        super().__init__(curricula)

    def get_task(self) -> Task:
        task_id = random.choices(list(self._curricula.keys()), weights=list(self._task_weights.values()))[0]
        task = self._curricula[task_id].get_task()
        task.add_parent(self, task_id)
        logger.debug(f"Task selected: {task.name()}")
        return task

    def _curriculum_from_id(self, cfg_path: str) -> Curriculum:
        return curriculum_from_config_path(cfg_path, self.env_overrides)
