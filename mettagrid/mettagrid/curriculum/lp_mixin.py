from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from omegaconf import DictConfig

from .learning_progress import BidirectionalLearningProgess

logger = logging.getLogger(__name__)


class LearningProgressMixin:
    """
    Mixin that adds learning progress capabilities to any curriculum.

    Usage:
        class MyCurriculum(LearningProgressMixin, BaseCurriculum):
            def __init__(self, *args, lp_weight: float = 0.0, **kwargs):
                super().__init__(*args, **kwargs)
                self._init_lp(lp_weight)
    """

    def _init_lp(self, lp_weight: float = 0.0, **lp_kwargs):
        """Initialize learning progress tracking."""
        self.lp_weight = lp_weight
        self.lp_enabled = lp_weight > 0.0

        if self.lp_enabled:
            # Determine search space size based on curriculum type
            if hasattr(self, '_curriculums'):
                # Multi-task curriculum
                search_space_size = len(self._curriculums)
            elif hasattr(self, '_cfg_template'):
                # Single task with parameter space
                search_space_size = 1
            else:
                # Fallback
                search_space_size = 1

            self.lp_tracker = BidirectionalLearningProgess(
                search_space=search_space_size,
                **lp_kwargs
            )
            logger.info(f"Learning progress enabled with weight {lp_weight}")
        else:
            self.lp_tracker = None
            logger.info("Learning progress disabled")

    def _update_weights_with_lp(self, base_weights: Dict[str, float], task_id: str, score: float) -> Dict[str, float]:
        """Update weights by blending base weights with learning progress weights."""
        if not self.lp_enabled or self.lp_tracker is None:
            return base_weights

        # Collect learning progress data
        if hasattr(self, '_curriculums'):
            # Multi-task: use task_id as index
            task_idx = list(self._curriculums.keys()).index(task_id)
            self.lp_tracker.collect_data({f'tasks/{task_idx}': [score]})
        else:
            # Single task: use 0 as index
            self.lp_tracker.collect_data({f'tasks/0': [score]})

        # Get learning progress weights
        lp_weights, _ = self.lp_tracker.calculate_dist()
        if lp_weights is None:
            return base_weights

        # Blend weights
        blended_weights = {}
        if hasattr(self, '_curriculums'):
            # Multi-task curriculum
            for i, task_id in enumerate(self._curriculums.keys()):
                base_weight = base_weights.get(task_id, 1.0)
                lp_weight = lp_weights[i] if i < len(lp_weights) else 1.0
                blended_weights[task_id] = (1 - self.lp_weight) * base_weight + self.lp_weight * lp_weight
        else:
            # Single task curriculum - return original weights
            blended_weights = base_weights

        # Normalize weights
        total_weight = sum(blended_weights.values())
        if total_weight > 0:
            blended_weights = {k: v / total_weight for k, v in blended_weights.items()}

        return blended_weights

    def _get_lp_stats(self) -> Dict[str, float]:
        """Get learning progress statistics for logging."""
        if not self.lp_enabled or self.lp_tracker is None:
            return {}

        stats = {}
        self.lp_tracker.add_stats(stats)
        return {f"lp/{k}": v for k, v in stats.items()}
