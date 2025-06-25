from __future__ import annotations

import copy
import logging
from typing import Dict, Optional

from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.core import Task
from metta.mettagrid.curriculum.random import RandomCurriculum
from metta.mettagrid.curriculum.sampling import SamplingCurriculum

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


class ProgressiveMultiTaskCurriculum(RandomCurriculum):
    """Curriculum that starts with higher probabilities for earlier tasks
    and gradually shifts to favor later tasks over time."""

    def __init__(
        self,
        tasks: Dict[str, float],
        env_overrides: DictConfig,
        progression_rate: float = 0.00001,
        initial_skew: float = 5.0,
    ):
        super().__init__(tasks, env_overrides)
        self._task_order = list(tasks.keys())  # Preserve order from dict
        self._progression_rate = progression_rate  # How fast to shift probabilities
        self._initial_skew = initial_skew  # How much to favor early tasks initially
        self._step_count = 0

        # Initialize weights heavily skewed toward beginning
        self._update_progressive_weights()

    def _update_progressive_weights(self):
        """Update task weights based on progression through training."""
        num_tasks = len(self._task_order)

        # Create a progression factor that goes from 0 to 1 over time
        progression = min(1.0, self._step_count * self._progression_rate)

        # Generate weights that start favoring early tasks and shift to later ones
        weights = {}
        for i, task_id in enumerate(self._task_order):
            # Position in list (0 to 1)
            position = i / (num_tasks - 1) if num_tasks > 1 else 0

            # Early in training: favor early tasks (low position values)
            # Later in training: favor later tasks (high position values)
            early_weight = self._initial_skew * (1 - position)  # Higher for early tasks
            late_weight = self._initial_skew * position  # Higher for later tasks

            # Interpolate between early and late weights based on progression
            weight = (1 - progression) * early_weight + progression * late_weight
            weights[task_id] = max(0.01, weight)  # Ensure minimum weight

        self._task_weights = weights

        logger.debug(
            f"Step {self._step_count}, progression: {progression:.3f}, "
            f"weights: {[(k, f'{v:.3f}') for k, v in weights.items()]}"
        )

    def complete_task(self, id: str, score: float):
        """Update step count and progressive weights after each task completion."""
        self._step_count += 1
        self._update_progressive_weights()
        super().complete_task(id, score)
