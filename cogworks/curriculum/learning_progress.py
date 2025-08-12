"""Learning progress curriculum implementation."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from pydantic import Field

from cogworks.curriculum.curriculum import Curriculum, PopulationCurriculumConfig
from cogworks.curriculum.task import Task

logger = logging.getLogger(__name__)


class LearningProgressCurriculumConfig(PopulationCurriculumConfig):
    """Configuration for LearningProgressCurriculum."""

    ema_timescale: float = Field(default=0.001, gt=0, le=1.0, description="EMA timescale for learning progress")
    progress_smoothing: float = Field(default=0.05, ge=0, le=1.0, description="Progress smoothing factor")
    rand_task_rate: float = Field(default=0.25, ge=0, le=1.0, description="Rate of random task selection")
    sample_threshold: int = Field(default=10, gt=0, description="Minimum samples before task becomes active")
    memory: int = Field(default=25, gt=0, description="Number of recent outcomes to remember per task")


class LearningProgressCurriculum(Curriculum):
    """LearningProgressCurriculum generates N tasks upfront and then samples based on learning progress."""

    def __init__(self, config: LearningProgressCurriculumConfig, seed: int = 0):
        super().__init__(config, seed)
        self._config: LearningProgressCurriculumConfig = config

        # Initialize empty task list - will be populated on first get_task call
        self._tasks = []
        self._tasks_initialized = False
        self._task_outcomes = {task.get_id(): [] for task in self._tasks}

        # Learning progress tracking state
        self._p_fast: np.ndarray | None = None
        self._p_slow: np.ndarray | None = None
        self._p_true: np.ndarray | None = None
        self._random_baseline: np.ndarray | None = None
        self._task_weights = np.ones(len(self._tasks)) / len(self._tasks)
        self._active_task_indices = list(range(min(self._config.num_active_tasks, len(self._tasks))))

    def _generate_n_tasks(self, base_seed: int) -> List[Task]:
        """Generate N tasks using different seeds."""
        tasks = []
        for i in range(self._config.n_tasks):
            seed = base_seed + i
            env_cfg = self._task_generator.get_task(seed)
            task = Task(task_id=f"lp_{seed}", env_cfg=env_cfg)
            tasks.append(task)
        return tasks

    def get_task(self, seed: int) -> Task:
        """Sample task based on learning progress using the provided seed."""
        # Initialize tasks on first call
        if not self._tasks_initialized:
            self._tasks = self._generate_n_tasks(seed)
            self._task_outcomes = {task.get_id(): [] for task in self._tasks}
            self._task_weights = np.ones(len(self._tasks)) / len(self._tasks)
            self._active_task_indices = list(range(min(self._config.num_active_tasks, len(self._tasks))))
            self._tasks_initialized = True

        # Use internal RNG for consistent sampling behavior
        if not self._active_task_indices:
            # If no active tasks, select randomly
            task_idx = self._rng.randint(0, len(self._tasks) - 1)
        else:
            # Sample from active tasks based on weights
            active_weights = [self._task_weights[i] for i in self._active_task_indices]
            total_weight = sum(active_weights)

            if total_weight <= 0:
                # Uniform sampling if weights are invalid
                task_idx = self._rng.choice(self._active_task_indices)
            else:
                # Weighted sampling
                normalized_weights = [w / total_weight for w in active_weights]
                selected_idx = self._rng.choices(self._active_task_indices, weights=normalized_weights)[0]
                task_idx = selected_idx

        return self._tasks[task_idx]

    def complete_task(self, task: Task, score: float):
        """Update learning progress based on completed task."""
        task_id = task.get_id()

        if task_id not in self._task_outcomes:
            logger.warning(f"Unknown task completed: {task_id}")
            return

        # Store outcome
        self._task_outcomes[task_id].append(max(0.0, min(1.0, score)))

        # Keep only recent outcomes
        self._task_outcomes[task_id] = self._task_outcomes[task_id][-self._config.memory :]

        # Update learning progress
        self._update_learning_progress()

    def _update_learning_progress(self):
        """Update learning progress tracking."""
        # Calculate success rates
        success_rates = []
        for task in self._tasks:
            outcomes = self._task_outcomes[task.get_id()]
            if outcomes:
                success_rates.append(np.mean(outcomes))
            else:
                success_rates.append(0.0)

        success_rates = np.array(success_rates)

        # Initialize baseline if needed
        if self._random_baseline is None:
            self._random_baseline = np.minimum(success_rates, 0.75)  # Cap at 75%

        # Normalize success rates
        denominator = 1.0 - self._random_baseline
        denominator = np.where(denominator <= 0, 1.0, denominator)
        normalized_rates = np.maximum(success_rates - self._random_baseline, 0.0) / denominator

        # Update EMA trackers
        if self._p_fast is None:
            self._p_fast = normalized_rates.copy()
            self._p_slow = normalized_rates.copy()
            self._p_true = success_rates.copy()
        else:
            self._p_fast = normalized_rates * self._config.ema_timescale + self._p_fast * (
                1.0 - self._config.ema_timescale
            )
            self._p_slow = self._p_fast * self._config.ema_timescale + self._p_slow * (1.0 - self._config.ema_timescale)
            self._p_true = success_rates * self._config.ema_timescale + self._p_true * (
                1.0 - self._config.ema_timescale
            )

        # Calculate learning progress
        learning_progress = np.abs(self._reweight(self._p_fast) - self._reweight(self._p_slow))

        # Update task weights based on learning progress
        if np.std(learning_progress) > 0:
            normalized_lp = (learning_progress - np.mean(learning_progress)) / np.std(learning_progress)
            weights = self._sigmoid(normalized_lp)
            weights = weights / np.sum(weights)
            self._task_weights = weights
        else:
            self._task_weights = np.ones(len(self._tasks)) / len(self._tasks)

        # Update active tasks
        self._update_active_tasks(learning_progress)

    def _reweight(self, probs: np.ndarray) -> np.ndarray:
        """Apply progress smoothing reweighting."""
        numerator = probs * (1.0 - self._config.progress_smoothing)
        denominator = probs + self._config.progress_smoothing * (1.0 - 2.0 * probs)
        denominator = np.where(denominator <= 0, 1.0, denominator)
        return numerator / denominator

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

    def _update_active_tasks(self, learning_progress: np.ndarray):
        """Update the set of active tasks based on learning progress."""
        # Select top tasks by learning progress, plus some random ones
        n_progress_tasks = int(self._config.num_active_tasks * (1.0 - self._config.rand_task_rate))
        n_random_tasks = self._config.num_active_tasks - n_progress_tasks

        # Top tasks by learning progress
        top_indices = np.argsort(learning_progress)[-n_progress_tasks:].tolist()

        # Random tasks
        all_indices = set(range(len(self._tasks)))
        remaining_indices = all_indices - set(top_indices)
        if remaining_indices and n_random_tasks > 0:
            random_indices = self._rng.sample(list(remaining_indices), min(n_random_tasks, len(remaining_indices)))
        else:
            random_indices = []

        self._active_task_indices = top_indices + random_indices

    def get_task_probs(self) -> dict[str, float]:
        """Return current task probabilities for logging."""
        probs = {}
        for i, task in enumerate(self._tasks):
            if i in self._active_task_indices:
                probs[task.get_id()] = float(self._task_weights[i])
            else:
                probs[task.get_id()] = 0.0
        return probs

    def get_curriculum_stats(self) -> dict:
        """Return learning progress statistics."""
        if self._p_fast is None:
            return {}

        stats = {
            "lp/num_active_tasks": len(self._active_task_indices),
            "lp/mean_task_weight": float(np.mean(self._task_weights)),
            "lp/num_zero_weight_tasks": int(np.sum(self._task_weights == 0)),
        }

        # Success rate statistics
        success_rates = []
        for task in self._tasks:
            outcomes = self._task_outcomes[task.get_id()]
            if outcomes:
                success_rates.append(np.mean(outcomes))
            else:
                success_rates.append(0.0)

        if success_rates:
            stats.update(
                {
                    "lp/mean_success_rate": float(np.mean(success_rates)),
                    "lp/min_success_rate": float(np.min(success_rates)),
                    "lp/max_success_rate": float(np.max(success_rates)),
                }
            )

        return stats
