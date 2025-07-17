from __future__ import annotations

import logging
import random
from typing import Dict

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

    def get_task(self) -> Task:
        task_id = random.choices(list(self._curricula.keys()), weights=list(self._task_weights.values()))[0]
        task = self._curricula[task_id].get_task()
        task.add_parent(self, task_id)
        logger.debug(f"Task selected: {task.name()}")
        return task

    def _curriculum_from_id(self, cfg_path: str) -> Curriculum:
        return curriculum_from_config_path(cfg_path, self.env_overrides)

    def get_env_cfg_by_bucket(self) -> dict[str, DictConfig]:
        configs = {}
        for task_id, curriculum in self._curricula.items():
            # Get configs from child curriculums without creating tasks
            child_configs = curriculum.get_env_cfg_by_bucket()
            # Use task_id as key if child returns a single config
            if len(child_configs) == 1:
                configs[task_id] = list(child_configs.values())[0]
            else:
                # Prefix multiple configs with task_id
                for sub_id, cfg in child_configs.items():
                    configs[f"{task_id}/{sub_id}"] = cfg
        return configs
