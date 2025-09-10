"""
Task Type Learning Progress Algorithm

Simplified learning progress that operates on task types instead of individual tasks.
This provides O(1) performance by maintaining EMA scores per task type.
"""

from typing import Dict

import numpy as np

from .learning_progress_algorithm import LearningProgressConfig


class TaskTypeLearningProgress:
    """Learning progress algorithm that operates on task types for O(1) performance."""

    def __init__(self, config: LearningProgressConfig, task_types: list[str]):
        self._config = config
        self._task_types = task_types

        # Task type EMA tracking
        self._task_type_emas: Dict[str, float] = {task_type: 0.0 for task_type in task_types}
        self._task_type_counts: Dict[str, int] = {task_type: 0 for task_type in task_types}

        # Simple mapping from task_id to task_type for performance updates
        self._task_id_to_type: Dict[int, str] = {}

    def register_task(self, task_id: int, task_type: str) -> None:
        """Register which task type a task_id belongs to."""
        self._task_id_to_type[task_id] = task_type

    def update_task_performance(self, task_id: int, score: float, bucket_values: Dict = None, curriculum=None) -> None:
        """Update EMA for the task type and trigger weight updates."""
        task_type = self._task_id_to_type.get(task_id)
        if not task_type:
            return

        # Update EMA for this task type
        alpha = self._config.ema_timescale
        current_ema = self._task_type_emas[task_type]
        self._task_type_emas[task_type] = (1 - alpha) * current_ema + alpha * score
        self._task_type_counts[task_type] += 1

        # Calculate learning progress scores (how much improvement)
        task_type_scores = {}
        for ttype in self._task_types:
            ema_score = self._task_type_emas[ttype]
            # Higher EMA = better performance, but we want learning progress
            # Use exploration bonus for types with low sample counts
            count = max(1, self._task_type_counts[ttype])
            exploration_bonus = self._config.exploration_bonus / np.sqrt(count)

            # Learning progress = potential improvement + exploration bonus
            progress_score = (1.0 - ema_score) + exploration_bonus
            task_type_scores[ttype] = max(0.001, progress_score)

        # Update curriculum weights
        if curriculum and hasattr(curriculum, "update_task_type_weights"):
            curriculum.update_task_type_weights(task_type_scores)

    def get_stats(self) -> Dict[str, float]:
        """Get statistics about task type performance."""
        stats = {}
        for task_type in self._task_types:
            stats[f"task_type_{task_type}_ema"] = self._task_type_emas[task_type]
            stats[f"task_type_{task_type}_count"] = float(self._task_type_counts[task_type])
        return stats
