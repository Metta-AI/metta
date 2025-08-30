"""Learning progress curriculum implementation."""

from __future__ import annotations

import logging

from pydantic import Field

from .curriculum import CurriculumConfig

logger = logging.getLogger(__name__)


class LearningProgressCurriculumConfig(CurriculumConfig):
    """Configuration for LearningProgressCurriculum."""

    ema_timescale: float = Field(default=0.001, gt=0, le=1.0, description="EMA timescale for learning progress")
    progress_smoothing: float = Field(default=0.05, ge=0, le=1.0, description="Progress smoothing factor")
    rand_task_rate: float = Field(default=0.25, ge=0, le=1.0, description="Rate of random task selection")
    memory: int = Field(default=25, gt=0, description="Number of recent outcomes to remember per task")


# TODO #dehydration
# class LearningProgressCurriculumTask(CurriculumTask):
#     """CurriculumTask that tracks learning progress internally."""

#     def __init__(self, config: LearningProgressCurriculumConfig, task_id: int, env_cfg: MettaGridConfig):
#         super().__init__(task_id, env_cfg)
#         self._config: LearningProgressCurriculumConfig = config

#         # Learning progress tracking state for this task
#         self._outcomes: List[float] = []
#         self._p_fast: float | None = None
#         self._p_slow: float | None = None
#         self._p_true: float | None = None
#         self._random_baseline: float | None = None

#     def complete(self, score: float):
#         """Complete task and update learning progress tracking."""
#         # Call parent complete() to update base statistics
#         super().complete(score)

#         # Store clipped outcome
#         clipped_score = max(0.0, min(1.0, score))
#         self._outcomes.append(clipped_score)

#         # Respect memory limit
#         memory_limit = self._config.memory
#         if len(self._outcomes) > memory_limit:
#             self._outcomes = self._outcomes[-memory_limit:]

#         # Update learning progress trackers
#         self._update_learning_progress()

#     def _update_learning_progress(self):
#         """Update learning progress tracking for this task."""
#         if not self._outcomes:
#             return

#         success_rate = np.mean(self._outcomes)

#         # Initialize baseline on first completion
#         if self._random_baseline is None:
#             self._random_baseline = min(success_rate, 0.75)  # Cap at 75%

#         # Normalize success rate
#         denominator = 1.0 - self._random_baseline
#         if denominator <= 0:
#             denominator = 1.0
#         normalized_rate = max(success_rate - self._random_baseline, 0.0) / denominator

#         # Update EMA trackers
#         ema_timescale = self._config.ema_timescale
#         if self._p_fast is None:
#             # Initialize trackers
#             self._p_fast = normalized_rate
#             self._p_slow = normalized_rate
#             self._p_true = success_rate
#         else:
#             # Update trackers
#             self._p_fast = normalized_rate * ema_timescale + self._p_fast * (1.0 - ema_timescale)
#             self._p_slow = self._p_fast * ema_timescale + self._p_slow * (1.0 - ema_timescale)
#             self._p_true = success_rate * ema_timescale + self._p_true * (1.0 - ema_timescale)

#     def get_learning_progress(self) -> float:
#         """Calculate and return current learning progress for this task."""
#         if self._p_fast is None or self._p_slow is None:
#             return 0.0

#         progress_smoothing = self._config.progress_smoothing

#         # Apply reweighting
#         reweighted_fast = self._reweight(self._p_fast, progress_smoothing)
#         reweighted_slow = self._reweight(self._p_slow, progress_smoothing)

#         return abs(reweighted_fast - reweighted_slow)

#     def _reweight(self, prob: float, progress_smoothing: float) -> float:
#         """Apply progress smoothing reweighting to a single probability."""
#         numerator = prob * (1.0 - progress_smoothing)
#         denominator = prob + progress_smoothing * (1.0 - 2.0 * prob)
#         if denominator <= 0:
#             denominator = 1.0
#         return numerator / denominator

#     def get_success_rate(self) -> float:
#         """Get current success rate for this task."""
#         if not self._outcomes:
#             return 0.0
#         return float(np.mean(self._outcomes))

#     def get_task_stats(self) -> dict:
#         """Get statistics for this specific task."""
#         return {
#             'success_rate': self.get_success_rate(),
#             'num_completions': self._num_completions,
#             'learning_progress': self.get_learning_progress(),
#             'num_outcomes': len(self._outcomes)
#         }


# class LearningProgressCurriculum(Curriculum):
#     """LearningProgressCurriculum samples tasks based on learning progress stored in each task."""

#     def __init__(self, config: LearningProgressCurriculumConfig, seed: int = 0):
#         super().__init__(config, seed)
#         self._config: LearningProgressCurriculumConfig = config
#         self._learning_progress_tasks: List[LearningProgressCurriculumTask] = []
#         self._task_weights: np.ndarray | None = None
#         self._active_task_indices: List[int] = []

#     def _choose_task(self) -> LearningProgressCurriculumTask:
#         self._update_task_weights()

#         if not self._active_task_indices:
#             # If no active tasks, select randomly
#             task_idx = self._rng.randint(0, len(self._learning_progress_tasks) - 1)
#             return self._learning_progress_tasks[task_idx]

#         # Weighted sampling from active tasks
#         active_weights = [self._task_weights[i] for i in self._active_task_indices]
#         total_weight = sum(active_weights)

#         if total_weight <= 0:
#             # Uniform sampling if weights are invalid
#             task_idx = self._rng.choice(self._active_task_indices)
#         else:
#             # Weighted sampling
#             normalized_weights = [w / total_weight for w in active_weights]
#             selected_idx = self._rng.choices(self._active_task_indices, weights=normalized_weights)[0]
#             task_idx = selected_idx

#         return self._learning_progress_tasks[task_idx]

#     def _create_task(self) -> LearningProgressCurriculumTask:
#         """Create a new LearningProgressCurriculumTask."""
#         # Use base class _create_task to get basic task creation
#         base_task = super()._create_task()

#         # Create LearningProgressCurriculumTask with same parameters
#         lp_task = LearningProgressCurriculumTask(self._config, base_task._task_id, base_task._env_cfg)

#         # Add to our learning progress task list
#         self._learning_progress_tasks.append(lp_task)

#         # Remove the base task from the tasks dict since we're using our own list
#         del self._tasks[base_task._task_id]

#         return lp_task

#     def _evict_task(self):
#         """Evict a learning progress task from the population."""
#         if not self._learning_progress_tasks:
#             return

#         # Choose task to evict
#         task_to_evict = self._rng.choice(self._learning_progress_tasks)

#         # Remove from our list
#         self._learning_progress_tasks.remove(task_to_evict)

#         # Remove from task_ids tracking set
#         if task_to_evict._task_id in self._task_ids:
#             self._task_ids.remove(task_to_evict._task_id)

#         self._num_evicted += 1

#     def _update_task_weights(self):
#         """Update task weights based on learning progress of each task."""
#         if not self._learning_progress_tasks:
#             self._task_weights = np.array([])
#             self._active_task_indices = []
#             return

#         # Get learning progress from each task
#         learning_progress = np.array([task.get_learning_progress() for task in self._learning_progress_tasks])

#         # Calculate weights based on learning progress
#         if np.std(learning_progress) > 0:
#             normalized_lp = (learning_progress - np.mean(learning_progress)) / np.std(learning_progress)
#             weights = self._sigmoid(normalized_lp)
#             weights = weights / np.sum(weights)
#             self._task_weights = weights
#         else:
#             # Uniform weights if no variation in learning progress
#             self._task_weights = np.ones(len(self._learning_progress_tasks)) / len(self._learning_progress_tasks)

#         # Update active tasks based on learning progress
#         self._update_active_tasks(learning_progress)

#     def _sigmoid(self, x: np.ndarray) -> np.ndarray:
#         """Apply sigmoid function."""
#         return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

#     def _update_active_tasks(self, learning_progress: np.ndarray):
#         """Update the set of active tasks based on learning progress."""
#         n_tasks = len(self._learning_progress_tasks)
#         if n_tasks == 0:
#             self._active_task_indices = []
#             return

#         # Determine how many tasks to make active
#         n_active = min(self._config.num_active_tasks, n_tasks)

#         # Select top tasks by learning progress, plus some random ones
#         n_progress_tasks = int(n_active * (1.0 - self._config.rand_task_rate))
#         n_random_tasks = n_active - n_progress_tasks

#         # Top tasks by learning progress
#         if n_progress_tasks > 0:
#             top_indices = np.argsort(learning_progress)[-n_progress_tasks:].tolist()
#         else:
#             top_indices = []

#         # Random tasks
#         all_indices = set(range(n_tasks))
#         remaining_indices = all_indices - set(top_indices)
#         if remaining_indices and n_random_tasks > 0:
#             random_indices = self._rng.sample(list(remaining_indices), min(n_random_tasks, len(remaining_indices)))
#         else:
#             random_indices = []

#         self._active_task_indices = top_indices + random_indices

#     def get_task_probs(self) -> dict[int, float]:
#         """Return current task probabilities for logging."""
#         probs = {}
#         for i, task in enumerate(self._learning_progress_tasks):
#             if i in self._active_task_indices and self._task_weights is not None:
#                 probs[task._task_id] = float(self._task_weights[i])
#             else:
#                 probs[task._task_id] = 0.0
#         return probs

#     def get_curriculum_stats(self) -> dict:
#         """Return learning progress statistics."""
#         if not self._learning_progress_tasks:
#             return {"lp/num_active_tasks": 0}

#         # Collect learning progress and success rates from tasks
#         learning_progress = [task.get_learning_progress() for task in self._learning_progress_tasks]
#         success_rates = [task.get_success_rate() for task in self._learning_progress_tasks]

#         stats = {
#             "lp/num_active_tasks": len(self._active_task_indices),
#         }

#         if self._task_weights is not None:
#             stats["lp/mean_task_weight"] = float(np.mean(self._task_weights))
#             stats["lp/num_zero_weight_tasks"] = int(np.sum(self._task_weights == 0))

#         if learning_progress:
#             stats.update({
#                 "lp/mean_learning_progress": float(np.mean(learning_progress)),
#                 "lp/min_learning_progress": float(np.min(learning_progress)),
#                 "lp/max_learning_progress": float(np.max(learning_progress)),
#             })

#         if success_rates:
#             stats.update({
#                 "lp/mean_success_rate": float(np.mean(success_rates)),
#                 "lp/min_success_rate": float(np.min(success_rates)),
#                 "lp/max_success_rate": float(np.max(success_rates)),
#             })

#         return stats
