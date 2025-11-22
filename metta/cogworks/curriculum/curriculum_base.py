"""Base classes and shared structures for curriculum system.

This module contains abstract base classes (CurriculumAlgorithm, CurriculumAlgorithmConfig)
and shared data structures (CurriculumTask) that need to be available to both curriculum.py
and algorithm implementations without creating circular imports.
"""

from __future__ import annotations

import abc
from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional

from pydantic import ConfigDict, Field

from metta.cogworks.curriculum.stats import StatsLogger
from mettagrid.base_config import Config

if TYPE_CHECKING:
    from metta.cogworks.curriculum.curriculum import Curriculum


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

        # Ensure label is always a string (not None or other types)
        label = getattr(env_cfg, "label", "unknown")
        self._label = str(label) if label is not None else "unknown"

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

    def get_label(self) -> str:
        """Get the task label for per-label metrics.

        Returns:
            Always returns a string (never None). Defaults to "unknown" if no label is set.
        """
        return self._label


class CurriculumAlgorithmConfig(Config, ABC):
    """Hyperparameters for the CurriculumAlgorithm."""

    type: str = Field(description="Type of algorithm hyperparameters")
    initial_weights: Optional[list[float]] = None
    num_active_tasks: int = Field(default=64, description="Number of active tasks in the curriculum pool")

    @abc.abstractmethod
    def algorithm_type(self) -> str:
        """Return the algorithm type string used in configs."""
        pass

    @abc.abstractmethod
    def create(self, num_tasks: int) -> "CurriculumAlgorithm":
        """Create the curriculum algorithm with these hyperparameters.

        Args:
            num_tasks: Number of tasks the algorithm will manage

        Returns:
            Configured curriculum algorithm instance
        """
        pass

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )


class CurriculumAlgorithm(StatsLogger, ABC):
    """Abstract base class for curriculum algorithms.

    Curriculum algorithms score tasks, recommend evictions, and track performance.
    The Curriculum maintains the task pool and lifecycle, while algorithms provide guidance.
    """

    num_tasks: int
    hypers: CurriculumAlgorithmConfig
    task_tracker: Optional[Any] = None

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
        """Get algorithm state for checkpointing. Override in subclasses that have state."""
        return {"type": self.hypers.algorithm_type()}

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load algorithm state from checkpoint. Override in subclasses that have state."""
        pass

    def on_task_created(self, task: CurriculumTask) -> None:
        """Notification that a new task has been created. Override if needed."""
        pass

    def on_task_sampled(self, task_id: int) -> None:
        """Notification that a task has been sampled. Override if needed."""
        pass

    def set_curriculum_reference(self, curriculum: "Curriculum") -> None:
        """Set reference to curriculum for stats updates. Override if needed."""
        pass

    def reset_epoch_counters(self) -> None:
        """Reset per-epoch counters. Override if needed."""
        pass

    def get_task_lp_score(self, task_id: int) -> float:
        """Get learning progress score for a task.

        Returns:
            LP score, or 0.0 if not applicable
        """
        return 0.0

    def calculate_gini_coefficients(self) -> Dict[str, float]:
        """Calculate Gini coefficients for the curriculum (optional, expensive operation).

        Override in subclasses that support Gini calculations. This is called once
        per epoch from centralized stats reporting, not per-worker.

        Returns:
            Dictionary of Gini coefficient stats (empty dict by default)
        """
        return {}

    def get_evictions_this_epoch(self) -> Dict[str, int]:
        """Get per-label eviction counts for this epoch WITHOUT resetting (optional).

        Returns:
            Dictionary mapping label -> eviction count (empty dict by default)
        """
        return {}

    def get_and_reset_evictions_this_epoch(self) -> Dict[str, int]:
        """Get and reset per-label eviction counts for this epoch (optional).

        This should ONLY be called at epoch boundaries.

        Returns:
            Dictionary mapping label -> eviction count (empty dict by default)
        """
        return {}

    def get_and_reset_sampling_counts_this_epoch(self) -> Dict[str, int]:
        """Get and reset per-label sampling counts for this epoch (optional).

        Returns:
            Dictionary mapping label -> sampling count (empty dict by default)
        """
        return {}

    def should_evict_task(self, task_id: int, min_presentations: int) -> bool:
        """Check if a task should be evicted based on algorithm-specific criteria.

        Default implementation returns False (no eviction). Subclasses should override
        to implement their own eviction criteria.

        Args:
            task_id: The task to check
            min_presentations: Minimum number of task presentations before eviction

        Returns:
            True if task should be evicted
        """
        return False

    def __init__(
        self, num_tasks: int, hypers: Optional[CurriculumAlgorithmConfig] = None, initialize_weights: bool = True
    ):
        if num_tasks <= 0:
            raise ValueError(f"Number of tasks must be positive. num_tasks {num_tasks}")
        self.num_tasks = num_tasks
        self.hypers = hypers  # type: ignore[assignment]
        self.task_tracker = None
        StatsLogger.__init__(self)

    def get_base_stats(self) -> Dict[str, float]:
        """Get basic statistics. Override in subclasses."""
        return {}

    def get_detailed_stats(self) -> Dict[str, float]:
        """Get detailed statistics. Override in subclasses."""
        return {}
