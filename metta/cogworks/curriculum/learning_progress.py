"""Learning progress curriculum implementation following codebase patterns."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from pydantic import Field

from .curriculum import Curriculum, CurriculumConfig, CurriculumTask
from .task_generator import TaskGeneratorConfigUnion

logger = logging.getLogger(__name__)


class LearningProgressCurriculumConfig(CurriculumConfig):
    """Configuration for LearningProgressCurriculum following the discriminated union pattern."""

    # Override the task_generator_config to use our learning progress specific config
    task_generator_config: TaskGeneratorConfigUnion = Field(
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
        if len(self._outcomes) > self._config.memory:
            self._outcomes = self._outcomes[-self._config.memory :]

        # Update learning progress trackers
        self._update_learning_progress()

    def _update_learning_progress(self):
        """Update learning progress tracking for this task."""
        if not self._outcomes:
            return

        success_rate = float(np.mean(self._outcomes))

        if not self._initialized:
            self._p_fast = success_rate
            self._p_slow = success_rate
            self._initialized = True
        else:
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
        super().__init__(config, seed)
        self._config: LearningProgressCurriculumConfig = config
        self._learning_progress_tasks: List[LearningProgressCurriculumTask] = []
        self._task_weights: List[float] = []

    def _create_task(self) -> LearningProgressCurriculumTask:
        """Create a new LearningProgressCurriculumTask."""
        # Use base class _create_task to get basic task creation
        base_task = super()._create_task()

        # Create LearningProgressCurriculumTask with same parameters
        lp_task = LearningProgressCurriculumTask(self._config, base_task._task_id, base_task._env_cfg)

        # Add to our learning progress task list
        self._learning_progress_tasks.append(lp_task)

        # Remove the base task from the tasks dict since we're using our own list
        del self._tasks[base_task._task_id]

        return lp_task

    def _choose_task(self) -> LearningProgressCurriculumTask:
        """Choose a task based on learning progress."""
        self._update_task_weights()

        if not self._learning_progress_tasks:
            # If no tasks, create one
            return self._create_task()

        # Simple weighted sampling based on learning progress
        if self._task_weights:
            task_idx = self._rng.choices(range(len(self._learning_progress_tasks)), weights=self._task_weights)[0]
            return self._learning_progress_tasks[task_idx]
        else:
            return self._learning_progress_tasks[self._rng.randint(0, len(self._learning_progress_tasks))]

    def _evict_task(self):
        """Evict a learning progress task from the population."""
        if not self._learning_progress_tasks:
            return

        # Choose task to evict
        task_to_evict = self._rng.choice(self._learning_progress_tasks)

        # Remove from our list
        self._learning_progress_tasks.remove(task_to_evict)

        # Remove from task_ids tracking set
        if task_to_evict._task_id in self._task_ids:
            self._task_ids.remove(task_to_evict._task_id)

        self._num_evicted += 1

    def _update_task_weights(self):
        """Update task weights based on learning progress of each task."""
        if not self._learning_progress_tasks:
            self._task_weights = []
            return

        # Get learning progress for each task
        learning_progress = [task.get_learning_progress() for task in self._learning_progress_tasks]

        # Simple normalization
        total = sum(learning_progress) + 1e-6
        self._task_weights = [lp / total for lp in learning_progress]

    def get_curriculum_stats(self) -> dict[str, float]:
        """Return learning progress statistics."""
        if not self._learning_progress_tasks:
            return {"lp/num_active_tasks": 0.0}

        # Collect learning progress and success rates from tasks
        learning_progress = [task.get_learning_progress() for task in self._learning_progress_tasks]
        success_rates = [task.get_success_rate() for task in self._learning_progress_tasks]

        stats: dict[str, float] = {
            "lp/num_active_tasks": float(len(self._learning_progress_tasks)),
        }

        if self._task_weights:
            stats["lp/mean_task_weight"] = float(np.mean(self._task_weights))
            stats["lp/num_zero_weight_tasks"] = int(np.sum(np.array(self._task_weights) == 0))

        if learning_progress:
            stats["lp/mean_learning_progress"] = float(np.mean(learning_progress))
            stats["lp/min_learning_progress"] = float(np.min(learning_progress))
            stats["lp/max_learning_progress"] = float(np.max(learning_progress))

        if success_rates:
            stats["lp/mean_success_rate"] = float(np.mean(success_rates))
            stats["lp/min_success_rate"] = float(np.min(success_rates))
            stats["lp/max_success_rate"] = float(np.max(success_rates))

        return stats
