from __future__ import annotations

import logging
import random
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from metta.mettagrid.curriculum.core import Curriculum, Task

logger = logging.getLogger(__name__)


class SampledTaskCurriculum(Curriculum):
    """Curriculum that contains a single task, but the task is sampled from a distribution."""

    def __init__(
        self,
        task_id: str,
        task_cfg_template: DictConfig,
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


def _sample(dist: Any) -> Any:
    if isinstance(dist, dict):
        if "range" in dist:
            lo, hi = dist["range"]
            value = np.random.uniform(lo, hi)
            if dist.get("want_int", False):
                value = int(value)
    elif isinstance(dist, (list, ListConfig)):
        # use random.choice instead of np.random.choice, since the latter will convert numbers to numpy types.
        # we want native python types since we're going to be putting them in OmegaConfs, and OmegaConf doesn't support
        # numpy types
        value = random.choice(dist)
    else:
        assert isinstance(dist, (int, float, str)), f"Invalid distribution type: {type(dist)}"
        value = dist
    return value
