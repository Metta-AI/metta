"""Learning progress curriculum implementation."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from pydantic import Field

from .curriculum import Curriculum, CurriculumConfig, CurriculumTask
from .task_generator import AnyTaskGeneratorConfig

logger = logging.getLogger(__name__)


class LearningProgressCurriculumConfig(CurriculumConfig):
    """Configuration for LearningProgressCurriculum."""

    # Override the task_generator_config to use our learning progress specific config
    task_generator_config: AnyTaskGeneratorConfig = Field(
        description="TaskGenerator configuration for learning progress curriculum"
    )

    # Learning progress specific parameters
    ema_timescale: float = Field(default=0.001, gt=0, le=1.0, description="EMA timescale for learning progress")
    progress_smoothing: float = Field(default=0.05, ge=0, le=1.0, description="Progress smoothing factor")
    rand_task_rate: float = Field(default=0.25, ge=0, le=1.0, description="Rate of random task selection")
    memory: int = Field(default=25, gt=0, description="Number of recent outcomes to remember per task")

    def make(self) -> LearningProgressCurriculum:
        """Make a LearningProgressCurriculum from this configuration."""
        return LearningProgressCurriculum(self)


class LearningProgressCurriculumTask(CurriculumTask):
    """CurriculumTask that tracks learning progress internally."""

    def __init__(self, config: LearningProgressCurriculumConfig, task_id: int, env_cfg):
        super().__init__(task_id, env_cfg)
        self._config: LearningProgressCurriculumConfig = config
        self._outcomes: List[float] = []
        self._p_fast: float = 0.0
        self._p_slow: float = 0.0
        self._initialized: bool = False

    def complete(self, score: float):
        """Complete task and update learning progress tracking."""
        # Call parent complete() to update base statistics
        super().complete(score)

        # Store clipped outcome
        clipped_score = max(0.0, min(1.0, score))
        self._outcomes.append(clipped_score)

        # Respect memory limit
        memory_limit = self._config.memory
        if len(self._outcomes) > memory_limit:
            self._outcomes = self._outcomes[-memory_limit:]

        # Update learning progress trackers
        self._update_learning_progress()

    def _update_learning_progress(self):
        """Update learning progress tracking for this task."""
        if not self._outcomes:
            return

        success_rate = float(np.mean(self._outcomes))

        # Initialize on first completion
        if not self._initialized:
            self._p_fast = success_rate
            self._p_slow = success_rate
            self._initialized = True
            return

        # Update EMA trackers
        ema_timescale = self._config.ema_timescale
        self._p_fast = float(success_rate * ema_timescale + self._p_fast * (1.0 - ema_timescale))
        self._p_slow = float(self._p_fast * ema_timescale + self._p_slow * (1.0 - ema_timescale))

    def get_learning_progress(self) -> float:
        """Calculate and return current learning progress for this task."""
        if not self._initialized:
            return 0.0
        return abs(self._p_fast - self._p_slow)

    def get_success_rate(self) -> float:
        """Get current success rate for this task."""
        if not self._outcomes:
            return 0.0
        return float(np.mean(self._outcomes))


class LearningProgressCurriculum(Curriculum):
    """LearningProgressCurriculum samples tasks based on learning progress stored in each task."""

    def __init__(self, config: LearningProgressCurriculumConfig, seed: int = 0):
        # Create learning progress algorithm hypers
        from .learning_progress_algorithm import LearningProgressHypers

        lp_hypers = LearningProgressHypers(
            ema_timescale=config.ema_timescale,
            progress_smoothing=config.progress_smoothing,
            num_active_tasks=config.num_active_tasks,
            rand_task_rate=config.rand_task_rate,
            sample_threshold=10,  # Default value
            memory=config.memory,
        )

        # Update config to include algorithm hypers
        config.algorithm_hypers = lp_hypers

        super().__init__(config, seed)
        self._config: LearningProgressCurriculumConfig = config

    def _create_task(self) -> LearningProgressCurriculumTask:
        """Create a new LearningProgressCurriculumTask."""
        # Generate task_id and env_cfg using base class logic
        task_id = self._rng.randint(0, self._config.max_task_id)
        while task_id in self._task_ids:
            task_id = self._rng.randint(0, self._config.max_task_id)
        self._task_ids.add(task_id)
        env_cfg = self._task_generator.get_task(task_id)

        # Create LearningProgressCurriculumTask
        task = LearningProgressCurriculumTask(self._config, task_id, env_cfg)
        self._tasks[task_id] = task
        self._num_created += 1
        return task

    def _choose_task(self) -> CurriculumTask:
        """Choose a task based on learning progress using the integrated algorithm."""
        # Use the base class _choose_task which now uses the algorithm
        return super()._choose_task()

    def _evict_task(self):
        """Evict a task from the population."""
        # Use base class eviction logic
        super()._evict_task()

    def get_curriculum_stats(self) -> dict[str, float]:
        """Return learning progress statistics."""
        if not self._tasks:
            return {"lp/num_active_tasks": 0.0}

        # Collect learning progress and success rates from tasks
        learning_progress = []
        success_rates = []

        for task in self._tasks.values():
            if isinstance(task, LearningProgressCurriculumTask):
                learning_progress.append(task.get_learning_progress())
                success_rates.append(task.get_success_rate())

        stats: dict[str, float] = {
            "lp/num_active_tasks": float(len(self._tasks)),
        }

        if learning_progress:
            stats["lp/mean_learning_progress"] = float(np.mean(learning_progress))
            stats["lp/min_learning_progress"] = float(np.min(learning_progress))
            stats["lp/max_learning_progress"] = float(np.max(learning_progress))

        if success_rates:
            stats["lp/mean_success_rate"] = float(np.mean(success_rates))
            stats["lp/min_success_rate"] = float(np.min(success_rates))
            stats["lp/max_success_rate"] = float(np.max(success_rates))

        return stats

    def stats(self) -> dict:
        """Override base stats method to include learning progress statistics."""
        base_stats = super().stats()
        lp_stats = self.get_curriculum_stats()
        return {**base_stats, **lp_stats}
