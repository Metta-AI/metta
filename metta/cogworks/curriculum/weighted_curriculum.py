"""Weighted curriculum implementation matching pre-dehydration performance architecture."""

from __future__ import annotations

import random
from typing import Dict, List

from .curriculum import CurriculumAlgorithm, CurriculumTask
from .task_generator import TaskGenerator
from .learning_progress_algorithm import LearningProgressConfig
from .task_type_learning_progress import TaskTypeLearningProgress


class WeightedCurriculum:
    """Fast curriculum using weighted sampling over task types (pre-dehydration style).
    
    Instead of managing individual task instances, this curriculum:
    1. Maintains weights for different task types
    2. Uses O(1) weighted random sampling to select task types
    3. Generates fresh task instances from selected types
    4. Updates type weights based on learning progress
    """

    def __init__(self, task_generator: TaskGenerator, config: LearningProgressConfig, seed: int = 0):
        self._task_generator = task_generator
        self._config = config
        self._rng = random.Random(seed)
        
        # Get all possible task types from the generator
        self._task_types = self._task_generator.get_all_task_types()
        
        # Initialize simplified learning progress algorithm for task types
        self._algorithm = TaskTypeLearningProgress(config, self._task_types)
        
        # Initialize uniform weights for all task types
        self._task_type_weights = {task_type: 1.0 for task_type in self._task_types}
        self._normalize_weights()
        
        # Statistics
        self._num_created = 0
        self._task_type_counts = {task_type: 0 for task_type in self._task_types}

    def get_task(self) -> CurriculumTask:
        """Sample a task using O(1) weighted sampling over task types."""
        # O(1) weighted sampling - the key performance optimization!
        task_type = self._rng.choices(
            population=list(self._task_type_weights.keys()),
            weights=list(self._task_type_weights.values())
        )[0]
        
        # Generate fresh task instance from selected type
        task_id = self._rng.randint(0, 1000000)  # Unique ID for this instance
        env_cfg = self._task_generator.get_task_by_type(task_type, task_id)
        task = CurriculumTask(task_id, env_cfg)
        
        # Register this task with the learning progress algorithm
        self._algorithm.register_task(task_id, task_type)
        
        # Update statistics
        self._num_created += 1
        self._task_type_counts[task_type] += 1
        
        return task

    def update_task_performance(self, task_id: int, score: float, bucket_values: Dict = None) -> None:
        """Update learning progress and task type weights."""
        # Let the learning progress algorithm process the performance
        # It will call back to update_task_type_weights() when ready
        self._algorithm.update_task_performance(task_id, score, bucket_values, self)

    def update_task_type_weights(self, task_type_scores: Dict[str, float]) -> None:
        """Update weights for task types based on learning progress scores."""
        # Update weights based on learning progress
        for task_type, score in task_type_scores.items():
            if task_type in self._task_type_weights:
                # Higher learning progress score = higher weight
                self._task_type_weights[task_type] = max(0.001, score)  # Minimum weight
        
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total_weight = sum(self._task_type_weights.values())
        if total_weight > 0:
            for task_type in self._task_type_weights:
                self._task_type_weights[task_type] /= total_weight
        else:
            # Fallback to uniform if all weights are zero
            uniform_weight = 1.0 / len(self._task_type_weights)
            for task_type in self._task_type_weights:
                self._task_type_weights[task_type] = uniform_weight

    def stats(self) -> Dict[str, float]:
        """Return curriculum statistics."""
        stats = {
            "num_created": float(self._num_created),
            "num_task_types": float(len(self._task_types)),
        }
        
        # Add task type distribution stats
        for task_type, count in self._task_type_counts.items():
            stats[f"task_type_{task_type}_count"] = float(count)
            stats[f"task_type_{task_type}_weight"] = self._task_type_weights[task_type]
        
        # Add learning progress algorithm stats
        stats.update(self._algorithm.get_stats())
            
        return stats