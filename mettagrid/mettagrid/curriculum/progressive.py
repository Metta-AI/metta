from __future__ import annotations

import copy
import logging
from typing import Optional

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
        return Task(f"sample({self._cfg_template.sampling})", self, cfg)

    def complete_task(self, id: str, score: float):
        if score > 0.5:
            self._width = min(self._width * 2, 100)
            self._height = min(self._height * 2, 100)
        super().complete_task(id, score)

    def get_task_probs(self):
        # For progressive curriculum, only the current task is active (prob=1), others are 0
        probs = {k: 0.0 for k in self._task_weights}
        if hasattr(self, '_current_task') and self._current_task in probs:
            probs[self._current_task] = 1.0
        elif self._task_weights:
            # fallback: set first task to 1.0 if _current_task is not set
            first = next(iter(self._task_weights))
            probs[first] = 1.0
        return probs
