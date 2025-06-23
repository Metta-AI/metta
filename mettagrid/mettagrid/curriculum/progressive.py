from __future__ import annotations

import copy
import logging
from typing import Optional

import wandb
from omegaconf import DictConfig, OmegaConf

from .curriculum import Task
from .sampling import SamplingCurriculum

logger = logging.getLogger(__name__)


class ProgressiveCurriculum(SamplingCurriculum):
    def __init__(self, env_cfg_template: str, env_overrides: Optional[DictConfig] = None):
        super().__init__(env_cfg_template, env_overrides)
        self._width = 10
        self._height = 10

    def get_task(self) -> Task:
        cfg = OmegaConf.create(copy.deepcopy(self._cfg_template))
        cfg.game.map.width = self._width
        cfg.game.map.height = self._height
        OmegaConf.resolve(cfg)
        # Log probability for the single task (always 1.0)
        if wandb.run is not None:
            task_id = f"sample({self._cfg_template.sampling})"
            wandb.run.log({"curriculum/task_probs": {task_id: 1.0}}, commit=False)
        return Task(f"sample({self._cfg_template.sampling})", self, cfg)

    def complete_task(self, id: str, score: float):
        if score > 0.5:
            self._width = min(self._width * 2, 100)
            self._height = min(self._height * 2, 100)
        super().complete_task(id, score)
