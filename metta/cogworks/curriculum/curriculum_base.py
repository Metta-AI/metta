"""Base curriculum data structures and abstract interfaces.

Separated to avoid circular imports between curriculum orchestrator and
algorithm implementations.
"""

from __future__ import annotations

import abc
from abc import ABC
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import ConfigDict, Field

from metta.cogworks.curriculum.stats import SliceAnalyzer, StatsLogger
from mettagrid.base_config import Config


class CurriculumTask:
    """A task instance with a task_id and env_cfg."""

    def __init__(self, task_id: int, env_cfg, slice_values: Optional[Dict[str, Any]] = None):
        self._task_id = task_id
        self._env_cfg = env_cfg
        self._slice_values = slice_values or {}
        self._num_completions = 0
        self._total_score = 0.0
        self._mean_score = 0.0
        self._num_scheduled = 0

    def complete(self, score: float):
        """Complete the task with a score."""
        self._num_completions += 1
        self._total_score += score
        self._mean_score = self._total_score / self._num_completions

    def get_env_cfg(self):
        """Get the environment configuration for this task."""
        return self._env_cfg

    def get_slice_values(self):
        """Get the slice values that were used to generate this task."""
        return self._slice_values

    def get_bucket_values(self):
        """Get the slice values (backward compatibility alias)."""
        return self._slice_values


class CurriculumAlgorithmConfig(Config, ABC):
    """Hyperparameters for the CurriculumAlgorithm."""

    type: str = Field(description="Type of algorithm hyperparameters")
    initial_weights: Optional[list[float]] = None

    @abc.abstractmethod
    def algorithm_type(self) -> str:
        """Return the algorithm type string used in configs."""
        pass

    def create(self, num_tasks: int) -> "CurriculumAlgorithm":
        """Create the curriculum algorithm with these hyperparameters."""
        from metta.cogworks.curriculum.curriculum import DiscreteRandomCurriculum

        return DiscreteRandomCurriculum(num_tasks, self)

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )


class CurriculumAlgorithm(StatsLogger, ABC):
    """Abstract base class for curriculum algorithms."""

    num_tasks: int
    hypers: CurriculumAlgorithmConfig

    @abc.abstractmethod
    def score_tasks(self, task_ids: List[int]) -> Dict[int, float]:
        """Score tasks for selection purposes. Higher scores = more likely to be selected."""
        pass

    @abc.abstractmethod
    def recommend_eviction(self, task_ids: List[int]) -> Optional[int]:
        """Recommend which task to evict. Return None for random selection."""
        pass

    @abc.abstractmethod
    def on_task_evicted(self, task_id: int) -> None:
        """Notification that a task has been evicted from the pool."""
        pass

    @abc.abstractmethod
    def update_task_performance(self, task_id: int, score: float):
        """Update task performance. Override in subclasses that track performance."""
        pass

    def get_state(self) -> Dict[str, Any]:
        return {"type": self.hypers.algorithm_type()}

    def load_state(self, state: Dict[str, Any]) -> None:
        pass

    def on_task_created(self, task: CurriculumTask) -> None:
        pass

    def set_curriculum_reference(self, curriculum) -> None:
        pass

    def should_evict_task(self, task_id: int, min_presentations: int = 5) -> bool:
        return False

    def __init__(self, num_tasks: int, hypers: Optional[CurriculumAlgorithmConfig] = None):
        if num_tasks <= 0:
            raise ValueError(f"Number of tasks must be positive. num_tasks {num_tasks}")
        self.num_tasks = num_tasks
        if hypers is None:
            raise ValueError("CurriculumAlgorithmConfig must be provided")
        self.hypers = hypers

        enable_detailed = getattr(hypers, "enable_detailed_slice_logging", False)
        StatsLogger.__init__(self, enable_detailed_logging=enable_detailed)
        max_slice_axes = getattr(hypers, "max_slice_axes", 3)
        self.slice_analyzer = SliceAnalyzer(max_slice_axes=max_slice_axes, enable_detailed_logging=enable_detailed)

    def get_base_stats(self) -> Dict[str, float]:
        return {"num_tasks": self.num_tasks, **self.slice_analyzer.get_base_stats()}

    def get_detailed_stats(self) -> Dict[str, float]:
        return self.slice_analyzer.get_detailed_stats()

    def stats(self, prefix: str = "") -> dict[str, float]:
        return super().stats(prefix)
