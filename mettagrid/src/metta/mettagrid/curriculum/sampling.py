from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.core import Curriculum, Task
from metta.mettagrid.util.hydra import config_from_path

logger = logging.getLogger(__name__)


class SamplingCurriculum(Curriculum):
    def __init__(self, env_cfg_template: str, env_overrides: Optional[DictConfig] = None):
        self._cfg_template = config_from_path(env_cfg_template, env_overrides)
        self._num_completed_tasks = 0

    def get_task(self) -> Task:
        cfg = OmegaConf.create(copy.deepcopy(self._cfg_template))
        OmegaConf.resolve(cfg)
        return Task("sample", self, cfg)

    def get_curriculum_stats(self) -> dict:
        return {
            "task_prob/sample": 1.0,
            "task_completions/sample": self._num_completed_tasks,
        }


class SampledTaskCurriculum(Curriculum):
    """Curriculum that contains a single task, but the task is sampled from a distribution."""

    def __init__(
        self,
        task_id: str,
        task_cfg_template: OmegaConf.DictConfig,
        sampling_parameters: Dict[str, Dict[str, Any]],
    ):
        self._task_id = task_id
        self._task_cfg_template = task_cfg_template
        self._sampling_parameters = sampling_parameters

    def get_task(self) -> Task:
        cfg = self._task_cfg_template.copy()
        for k, v in self._sampling_parameters.items():
            OmegaConf.update(cfg, k, _sample(v), merge=False)
        return Task(self._task_id, self, cfg)


def _sample(choice: Any) -> Any:
    if isinstance(choice, dict) and "range" in choice:
        lo, hi = choice["range"]
        value = np.random.uniform(lo, hi)
        if choice.get("want_int", False):
            value = int(value)
        return value
    return choice
