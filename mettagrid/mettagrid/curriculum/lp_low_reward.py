from __future__ import annotations

import logging
from typing import Dict

from omegaconf import DictConfig

from .low_reward import LowRewardCurriculum
from .lp_mixin import LearningProgressMixin

logger = logging.getLogger(__name__)


class LPLowRewardCurriculum(LearningProgressMixin, LowRewardCurriculum):
    """Low reward curriculum enhanced with learning progress capabilities."""

    def __init__(self, tasks: Dict[str, float], env_overrides: DictConfig,
                 lp_weight: float = 0.0, **lp_kwargs):
        # Initialize base curriculum first
        super().__init__(tasks, env_overrides)
        # Initialize learning progress
        self._init_lp(lp_weight, **lp_kwargs)

    def complete_task(self, id: str, score: float):
        # First, update reward-based weights (original LowReward logic)
        old_average = self._reward_averages[id]
        self._reward_averages[id] = (1 - self._alpha) * self._reward_averages[id] + self._alpha * score
        self._reward_maxes[id] = max(self._reward_maxes[id], score)

        # Calculate reward-based weights
        reward_weights = {
            t: 1e-6 + self._reward_maxes[t] / (self._reward_averages[t] + 1e-6)
            for t in self._curriculums.keys()
        }

        # Blend with learning progress weights
        self._task_weights = self._update_weights_with_lp(reward_weights, id, score)

        # Log learning progress stats
        lp_stats = self._get_lp_stats()
        if lp_stats:
            logger.debug(f"LP stats: {lp_stats}")

        super().complete_task(id, score)
