from __future__ import annotations

import logging
import random
from typing import Dict

from omegaconf import DictConfig

from .curriculum import Task
from .multi_task import MultiTaskCurriculum
from .lp_mixin import LearningProgressMixin

logger = logging.getLogger(__name__)


class LPRandomCurriculum(LearningProgressMixin, MultiTaskCurriculum):
    """Random curriculum enhanced with learning progress capabilities."""

    def __init__(self, tasks: Dict[str, float], env_overrides: DictConfig,
                 lp_weight: float = 0.0, **lp_kwargs):
        # Initialize base curriculum first
        super().__init__(tasks, env_overrides)
        # Initialize learning progress
        self._init_lp(lp_weight, **lp_kwargs)

    def complete_task(self, id: str, score: float):
        # Update weights with learning progress
        self._task_weights = self._update_weights_with_lp(self._task_weights, id, score)

        # Log learning progress stats
        lp_stats = self._get_lp_stats()
        if lp_stats:
            logger.debug(f"LP stats: {lp_stats}")

        super().complete_task(id, score)
