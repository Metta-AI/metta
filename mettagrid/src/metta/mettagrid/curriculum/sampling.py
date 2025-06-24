from __future__ import annotations

import copy
import logging
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.util.hydra import config_from_path

from .core import Curriculum, Task

logger = logging.getLogger(__name__)


class SamplingCurriculum(Curriculum):
    def __init__(self, env_cfg_template: str, env_overrides: Optional[DictConfig] = None):
        self._cfg_template = config_from_path(env_cfg_template, env_overrides)

    def get_task(self) -> Task:
        cfg = OmegaConf.create(copy.deepcopy(self._cfg_template))
        OmegaConf.resolve(cfg)
        return Task(f"sample({self._cfg_template.sampling})", self, cfg)

    def get_task_probs(self) -> dict[str, float]:
        """Return the current task probability for logging purposes."""
        task_name = f"sample({self._cfg_template.sampling})"
        return {task_name: 1.0}
