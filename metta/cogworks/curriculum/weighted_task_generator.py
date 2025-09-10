"""Weighted task generator that wraps existing generators with weighted sampling."""

import random
from typing import Dict

from .learning_progress_algorithm import LearningProgressConfig
from .task_generator import TaskGenerator
from .task_type_learning_progress import TaskTypeLearningProgress


class WeightedTaskGenerator:
    """Task generator wrapper that uses weighted sampling over task types."""

    def __init__(self, base_generator: TaskGenerator, config: LearningProgressConfig, seed: int = 0):
        self._base_generator = base_generator
        self._config = config
        self._rng = random.Random(seed)

        # Get task types and initialize weights
        self._task_types = base_generator.get_all_task_types()
        self._task_type_weights = {task_type: 1.0 for task_type in self._task_types}
        self._task_type_counts = {task_type: 0 for task_type in self._task_types}

        # Learning progress for task types
        self._learning_progress = TaskTypeLearningProgress(config, self._task_types)

        # Task ID to type mapping for performance updates
        self._task_id_to_type: Dict[int, str] = {}

        self._normalize_weights()

    def get_task(self, task_id: int):
        """Generate a task using weighted sampling over task types."""
        if not self._task_types:
            # Fallback to base generator
            return self._base_generator.get_task(task_id)

        # Weighted sampling over task types
        task_type = self._rng.choices(
            population=list(self._task_type_weights.keys()), weights=list(self._task_type_weights.values())
        )[0]

        # Generate task from selected type
        env_cfg = self._base_generator.get_task_by_type(task_type, task_id)

        # Register this task for performance updates
        self._task_id_to_type[task_id] = task_type
        self._learning_progress.register_task(task_id, task_type)
        self._task_type_counts[task_type] += 1

        return env_cfg

    def get_all_task_types(self):
        """Get all task types from the base generator."""
        return self._base_generator.get_all_task_types()

    def get_task_by_type(self, task_type: str, task_id: int):
        """Generate a task of specific type."""
        return self._base_generator.get_task_by_type(task_type, task_id)

    def update_task_performance(self, task_id: int, score: float, bucket_values: Dict = None):
        """Update learning progress and task type weights."""
        self._learning_progress.update_task_performance(task_id, score, bucket_values, self)

    def update_task_type_weights(self, task_type_scores: Dict[str, float]) -> None:
        """Update weights for task types based on learning progress scores."""
        for task_type, score in task_type_scores.items():
            if task_type in self._task_type_weights:
                self._task_type_weights[task_type] = max(0.001, score)
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        if not self._task_type_weights:
            return

        total_weight = sum(self._task_type_weights.values())
        if total_weight > 0:
            for task_type in self._task_type_weights:
                self._task_type_weights[task_type] /= total_weight
        else:
            uniform_weight = 1.0 / len(self._task_type_weights)
            for task_type in self._task_type_weights:
                self._task_type_weights[task_type] = uniform_weight

    def get_stats(self) -> Dict[str, float]:
        """Get statistics about task type performance."""
        stats = {
            "num_task_types": float(len(self._task_types)),
        }

        for task_type, count in self._task_type_counts.items():
            stats[f"task_type_{task_type}_count"] = float(count)
            stats[f"task_type_{task_type}_weight"] = self._task_type_weights.get(task_type, 0.0)

        stats.update(self._learning_progress.get_stats())
        return stats

    # Delegate other methods to base generator
    def __getattr__(self, name):
        """Delegate unknown methods to base generator."""
        return getattr(self._base_generator, name)
