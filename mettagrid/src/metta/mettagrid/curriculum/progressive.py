from __future__ import annotations

import copy
import logging
from typing import Dict, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.core import Task
from metta.mettagrid.curriculum.random import RandomCurriculum
from metta.mettagrid.curriculum.sampling import SamplingCurriculum

logger = logging.getLogger(__name__)


class ProgressiveCurriculum(SamplingCurriculum):
    def __init__(self, env_cfg_template_path: str, env_overrides: Optional[DictConfig] = None):
        super().__init__(env_cfg_template_path, env_overrides)
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
    """Curriculum that blends multiple tasks using gating mechanisms and advances progression based on
    smoothed performance or time."""

    def __init__(
        self,
        tasks: Dict[str, float],
        env_overrides: Optional[DictConfig] = None,
        performance_threshold: float = 0.8,
        smoothing: float = 0.1,
        progression_rate: float = 0.01,
        progression_mode: str = "perf",
        blending_smoothness: float = 0.5,
        blending_mode: str = "logistic",
    ):
        if env_overrides is None:
            env_overrides = DictConfig({})
        super().__init__(tasks, env_overrides)
        if progression_mode not in ["time", "perf"]:
            raise ValueError("progression_mode must be either 'time' or 'perf'")
        if blending_mode not in ["logistic", "linear"]:
            raise ValueError("blending_mode must be either 'logistic' or 'linear'")
        self._task_order = list(tasks.keys())
        self._performance_threshold = performance_threshold
        self._smoothing = smoothing
        self._progression_rate = progression_rate
        self._progression_mode = progression_mode
        self._blending_smoothness = blending_smoothness
        self._blending_mode = blending_mode
        self._progress = 0.0  # initialization of the progress value parameterizing the trajectory
        self._smoothed_performance = 0.0
        self._step_count = 0
        self._last_score = None
        self._update_progressive_weights()

    def _update_smoothed_performance(self, score: float):
        if self._last_score is None:
            self._smoothed_performance = score
        else:
            self._smoothed_performance = self._smoothing * score + (1 - self._smoothing) * self._smoothed_performance
        self._last_score = score

    def _advance_progression(self):
        if self._progression_mode == "perf":
            if self._smoothed_performance >= self._performance_threshold:
                self._progress = min(1.0, self._progress + self._progression_rate)
        elif self._progression_mode == "time":
            self._step_count += 1
            self._progress = min(1.0, self._step_count * self._progression_rate)

    def _blending_function(self, x, xo, growing=True):
        """Blending function that supports both logistic and linear modes."""
        if self._blending_mode == "logistic":
            return 1 / (1 + np.exp(-((-1) ** growing) * (x - xo) / self._blending_smoothness))
        elif self._blending_mode == "linear":
            # Linear blending with smoothness control
            if growing:
                # Growing function: 0 at xo, 1 at xo + blending_smoothness
                return max(0, min(1, (x - xo + self._blending_smoothness) / self._blending_smoothness))
            else:
                # Shrinking function: 1 at xo, 0 at xo + blending_smoothness
                return max(0, min(1, (xo + self._blending_smoothness - x) / self._blending_smoothness))

    def _update_progressive_weights(self):
        num_tasks = len(self._task_order)

        # Use the gating mechanism for all progress values
        # Scale progress to task space (0 to num_tasks-1)
        p = self._progress * (num_tasks - 1)

        # Task positions (0, 1, 2, ..., num_tasks-1)
        task_positions = np.arange(num_tasks)

        # Create gating matrix: tasks x progress points
        gating = np.zeros(num_tasks)

        for i, task_pos in enumerate(task_positions):
            # Double gating: activation and deactivation
            activation = self._blending_function(p, task_pos - self._blending_smoothness, growing=True)
            deactivation = self._blending_function(p, task_pos + self._blending_smoothness, growing=False)
            gating[i] = activation * deactivation

        # Normalize to get probabilities
        if np.sum(gating) > 0:
            probs = gating / np.sum(gating)
        else:
            # Fallback to uniform distribution if all gates are zero
            probs = np.ones(num_tasks) / num_tasks

        self._task_weights = {task_id: float(probs[i]) for i, task_id in enumerate(self._task_order)}

        logger.debug(
            f"Progress: {self._progress:.3f}, smoothed_perf: {self._smoothed_performance:.3f}, "
            f"weights: {[(k, f'{v:.3f}') for k, v in self._task_weights.items()]}"
        )

    def complete_task(self, id: str, score: float):
        # Assume score is between 0 and 1
        self._update_smoothed_performance(score)
        self._advance_progression()
        self._update_progressive_weights()
        super().complete_task(id, score)

    def stats(self) -> Dict[str, float]:
        """Return curriculum statistics for logging purposes."""
        return {
            **super().stats(),
            "smoothed_performance": self._smoothed_performance,
            "progress": self._progress,
        }
