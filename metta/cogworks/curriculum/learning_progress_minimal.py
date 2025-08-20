"""Minimal learning progress curriculum implementation."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LearningProgressCurriculumConfig(BaseModel):
    """Minimal configuration for LearningProgressCurriculum."""

    ema_timescale: float = Field(default=0.001, gt=0, le=1.0)
    progress_smoothing: float = Field(default=0.05, ge=0, le=1.0)
    rand_task_rate: float = Field(default=0.25, ge=0, le=1.0)
    memory: int = Field(default=25, gt=0)


class LearningProgressCurriculumTask:
    """Minimal task that tracks learning progress."""

    def __init__(self, config: LearningProgressCurriculumConfig, task_id: int, env_cfg):
        self._config = config
        self._task_id = task_id
        self._env_cfg = env_cfg
        self._outcomes: List[float] = []
        self._p_fast: float = 0.0
        self._p_slow: float = 0.0
        self._initialized: bool = False

    def complete(self, score: float):
        """Complete task and update learning progress."""
        # Store clipped outcome
        clipped_score = max(0.0, min(1.0, score))
        self._outcomes.append(clipped_score)

        # Respect memory limit
        if len(self._outcomes) > self._config.memory:
            self._outcomes = self._outcomes[-self._config.memory :]

        # Update learning progress
        self._update_learning_progress()

    def _update_learning_progress(self):
        """Update learning progress tracking."""
        if not self._outcomes:
            return

        success_rate = float(np.mean(self._outcomes))

        if not self._initialized:
            self._p_fast = success_rate
            self._p_slow = success_rate
            self._initialized = True
        else:
            # Update EMA trackers
            self._p_fast = float(
                success_rate * self._config.ema_timescale + self._p_fast * (1.0 - self._config.ema_timescale)
            )
            self._p_slow = float(
                self._p_fast * self._config.ema_timescale + self._p_slow * (1.0 - self._config.ema_timescale)
            )

    def get_learning_progress(self) -> float:
        """Get current learning progress."""
        if not self._initialized:
            return 0.0
        return abs(self._p_fast - self._p_slow)

    def get_env_cfg(self):
        """Get environment configuration."""
        return self._env_cfg


class LearningProgressCurriculum:
    """Minimal learning progress curriculum."""

    def __init__(self, config: LearningProgressCurriculumConfig, seed: int = 0):
        self._config = config
        self._rng = np.random.RandomState(seed)
        self._tasks: List[LearningProgressCurriculumTask] = []
        self._task_weights: List[float] = []

    def get_task(self) -> LearningProgressCurriculumTask:
        """Get a task based on learning progress."""
        if not self._tasks:
            # Create initial task
            task = LearningProgressCurriculumTask(self._config, 0, {})
            self._tasks.append(task)
            return task

        # Simple weighted sampling based on learning progress
        if self._task_weights:
            task_idx = self._rng.choice(len(self._tasks), p=self._task_weights)
            return self._tasks[task_idx]
        else:
            return self._tasks[self._rng.randint(0, len(self._tasks))]

    def _update_weights(self):
        """Update task weights based on learning progress."""
        if not self._tasks:
            return

        # Get learning progress for each task
        learning_progress = [task.get_learning_progress() for task in self._tasks]

        # Simple normalization
        total = sum(learning_progress) + 1e-6
        self._task_weights = [lp / total for lp in learning_progress]

    def stats(self) -> dict:
        """Get curriculum statistics."""
        if not self._tasks:
            return {"lp/num_tasks": 0}

        learning_progress = [task.get_learning_progress() for task in self._tasks]
        return {
            "lp/num_tasks": len(self._tasks),
            "lp/mean_progress": float(np.mean(learning_progress)),
            "lp/max_progress": float(np.max(learning_progress)),
        }
