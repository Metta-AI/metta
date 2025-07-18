from __future__ import annotations

import logging
import random
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from metta.mettagrid.curriculum.curriculum import Curriculum, MettaGridTask, Task
from metta.mettagrid.util.hydra import config_from_path

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
        
        # Create a single task with the template config (will be resampled on each access)
        task = MettaGridTask(task_id, task_cfg_template)
        
        # Initialize parent Curriculum with a single task
        from metta.mettagrid.curriculum.curriculum_algorithm import DiscreteRandomHypers
        hypers = DiscreteRandomHypers()
        super().__init__(name=task_id, algorithm=hypers.create(1), tasks=[task])

    def sample(self) -> Task:
        # Create a new config with sampled parameters
        cfg = self._task_cfg_template.copy()
        for k, v in self._sampling_parameters.items():
            OmegaConf.update(cfg, k, _sample(v), merge=False)
        
        # Create a new task with the sampled config
        task = MettaGridTask(self._task_id, cfg)
        task.set_parent(self, 0)  # Single task, so always index 0
        return task


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
