from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.core import Curriculum, Task
from metta.mettagrid.util.hydra import config_from_path

logger = logging.getLogger(__name__)


class SamplingCurriculum(Curriculum):
    def __init__(self, env_cfg_template_path: str, env_overrides: Optional[DictConfig] = None):
        self._cfg_template = config_from_path(env_cfg_template_path, env_overrides)

    def get_task(self) -> Task:
        cfg = OmegaConf.create(copy.deepcopy(self._cfg_template))
        OmegaConf.resolve(cfg)
        return Task(f"sample({self._cfg_template.sampling})", self, cfg)

    def get_task_probs(self) -> dict[str, float]:
        """Return the current task probability for logging purposes."""
        task_name = f"sample({self._cfg_template.sampling})"
        return {task_name: 1.0}


class SampledTaskCurriculum(Curriculum):
    """Curriculum that contains a single task, but the task is sampled from a distribution."""

    def __init__(
        self,
        task_id: str,
        task_cfg_template: str,
        bucket_parameters: Dict[str, Dict[str, Any]],
        bucket_values: List[Any],
    ):
        self._task_id = task_id
        self._task_cfg_template = task_cfg_template
        self._bucket_values = bucket_values
        self._bucket_parameters = bucket_parameters
        self._completed_tasks = None

    def get_task(self) -> Task:
        cfg = self._task_cfg_template.copy()
        override = dict(zip(self._bucket_parameters, self._bucket_values, strict=False))
        for k, v in override.items():
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
