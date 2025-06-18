from __future__ import annotations

import logging
import random
from typing import Dict

from omegaconf import DictConfig

from mettagrid.curriculum.curriculum import Curriculum, Task
from mettagrid.curriculum.multi_task import MultiTaskCurriculum
from mettagrid.curriculum.util import curriculum_from_config_path

logger = logging.getLogger(__name__)


class RandomCurriculum(MultiTaskCurriculum):
    """Curriculum that samples from multiple environment types with fixed weights."""

    def __init__(self, curricula_cfgs: Dict[str, float], env_overrides: DictConfig):
        curricula = self.load_curricula(curricula_cfgs, env_overrides)
        super().__init__(curricula, env_overrides)

    def get_task(self) -> Task:
        task_id = random.choices(list(self._curriculums.keys()), weights=list(self._task_weights.values()))[0]
        task = self._curriculums[task_id].get_task()
        task.add_parent(self, task_id)
        logger.debug(f"Task selected: {task.name()}")
        return task

    def load_curricula(self, curricula_cfgs: Dict[str, float], env_overrides: DictConfig) -> Dict[str, Curriculum]:
        self._task_weights = curricula_cfgs
        return {t: curriculum_from_config_path(t, env_overrides) for t in curricula_cfgs.keys()}
