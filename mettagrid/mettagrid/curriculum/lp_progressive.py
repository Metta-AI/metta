from __future__ import annotations

import copy
import logging
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from .curriculum import Task
from .sampling import SamplingCurriculum
from .lp_mixin import LearningProgressMixin

logger = logging.getLogger(__name__)


class LPProgressiveCurriculum(LearningProgressMixin, SamplingCurriculum):
    """Progressive curriculum enhanced with learning progress capabilities."""

    def __init__(self, env_cfg_template: str, env_overrides: Optional[DictConfig] = None,
                 lp_weight: float = 0.0, **lp_kwargs):
        # Initialize base curriculum first
        super().__init__(env_cfg_template, env_overrides)
        # Initialize learning progress
        self._init_lp(lp_weight, **lp_kwargs)

        # Progressive curriculum state
        self._width = 10
        self._height = 10
        self._difficulty_levels = [10, 20, 40, 80, 100]
        self._current_level_idx = 0

    def get_task(self) -> Task:
        cfg = OmegaConf.create(copy.deepcopy(self._cfg_template))
        cfg.game.map.width = self._width
        cfg.game.map.height = self._height
        OmegaConf.resolve(cfg)
        return Task(f"sample({self._cfg_template.sampling})", self, cfg)

    def complete_task(self, id: str, score: float):
        if self.lp_enabled and self.lp_tracker is not None:
            # Use learning progress to determine progression
            self.lp_tracker.collect_data({f'tasks/0': [score]})
            lp_weights, _ = self.lp_tracker.calculate_dist()

            if lp_weights is not None:
                lp_score = lp_weights[0] if len(lp_weights) > 0 else 0.0

                # Low learning progress: stay or regress
                if lp_score < 0.1:
                    self._current_level_idx = max(0, self._current_level_idx - 1)
                    logger.debug(f"LP regression: level {self._current_level_idx}")
                # High performance: progress
                elif score > 0.5:
                    self._current_level_idx = min(len(self._difficulty_levels) - 1,
                                               self._current_level_idx + 1)
                    logger.debug(f"LP progression: level {self._current_level_idx}")

                self._width = self._height = self._difficulty_levels[self._current_level_idx]
            else:
                # Fallback to original logic
                if score > 0.5:
                    self._width = min(self._width * 2, 100)
                    self._height = min(self._height * 2, 100)
        else:
            # Original progressive logic when LP is disabled
            if score > 0.5:
                self._width = min(self._width * 2, 100)
                self._height = min(self._height * 2, 100)

        # Log learning progress stats
        lp_stats = self._get_lp_stats()
        if lp_stats:
            logger.debug(f"LP stats: {lp_stats}")

        super().complete_task(id, score)
