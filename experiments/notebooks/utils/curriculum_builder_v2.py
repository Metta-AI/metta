"""
Builder utilities for the new curriculum system (v2).

Provides a fluent API for building TaskSets and Curricula with the new architecture.
"""

from typing import Any, Dict, List, Union, Optional
from metta.mettagrid.mettagrid_config import EnvConfig
from .curriculum_v2 import (
    TaskSet,
    WeightedTaskSet,
    BucketedTaskSet,
    Curriculum,
    RandomCurriculum,
    LearningProgressCurriculum,
)


class TaskSetBuilder:
    """Fluent builder for creating TaskSets."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.items: List[tuple[Union[EnvConfig, TaskSet], float]] = []
        self.overrides: Dict[str, Any] = {}

    def add_config(self, config: EnvConfig, weight: float = 1.0) -> "TaskSetBuilder":
        """Add an environment configuration with weight."""
        self.items.append((config, weight))
        return self

    def add_task_set(self, task_set: TaskSet, weight: float = 1.0) -> "TaskSetBuilder":
        """Add another TaskSet with weight."""
        self.items.append((task_set, weight))
        return self

    def add_override(self, key: str, value: Any) -> "TaskSetBuilder":
        """Add a parameter override using dot notation."""
        self.overrides[key] = value
        return self

    def add_overrides(
        self, overrides: Union[Dict[str, Any], List[str]]
    ) -> "TaskSetBuilder":
        """Add multiple overrides from dict or list of 'key: value' strings."""
        if isinstance(overrides, dict):
            self.overrides.update(overrides)
        else:
            for override_str in overrides:
                if ":" not in override_str:
                    raise ValueError(
                        f"Override must be 'key: value', got: {override_str}"
                    )
                key, value = override_str.split(":", 1)
                self.add_override(key.strip(), value.strip())
        return self

    def build(self) -> WeightedTaskSet:
        """Build the WeightedTaskSet."""
        if not self.items:
            raise ValueError("Must add at least one config or task set")
        return WeightedTaskSet(self.items, self.overrides or None, self.seed)


class BucketedTaskSetBuilder:
    """Fluent builder for creating BucketedTaskSets."""

    def __init__(self, base_config: EnvConfig, seed: int = 42):
        self.base_config = base_config
        self.seed = seed
        self.buckets: Dict[str, Any] = {}
        self.overrides: Dict[str, Any] = {}

    def add_bucket_values(
        self, param_path: str, values: List[Any]
    ) -> "BucketedTaskSetBuilder":
        """Add a bucket with discrete values."""
        self.buckets[param_path] = values
        return self

    def add_bucket_range(
        self, param_path: str, min_val: float, max_val: float
    ) -> "BucketedTaskSetBuilder":
        """Add a bucket with a continuous range."""
        self.buckets[param_path] = {"range": [min_val, max_val]}
        return self

    def add_override(self, key: str, value: Any) -> "BucketedTaskSetBuilder":
        """Add a parameter override using dot notation."""
        self.overrides[key] = value
        return self

    def build(self) -> BucketedTaskSet:
        """Build the BucketedTaskSet."""
        if not self.buckets:
            raise ValueError("Must add at least one bucket")
        return BucketedTaskSet(
            self.base_config, self.buckets, self.overrides or None, self.seed
        )


class CurriculumBuilder:
    """Fluent builder for creating Curricula."""

    def __init__(self, task_set: TaskSet):
        self.task_set = task_set
        self.curriculum_type = "random"
        self.seed = 42
        self.num_tasks = 10  # For learning progress curriculum

    def as_random(self, seed: int = 42) -> "CurriculumBuilder":
        """Build as RandomCurriculum."""
        self.curriculum_type = "random"
        self.seed = seed
        return self

    def as_learning_progress(
        self, num_tasks: int = 10, seed: int = 42
    ) -> "CurriculumBuilder":
        """Build as LearningProgressCurriculum."""
        self.curriculum_type = "learning_progress"
        self.num_tasks = num_tasks
        self.seed = seed
        return self

    def build(self) -> Curriculum:
        """Build the curriculum."""
        if self.curriculum_type == "random":
            return RandomCurriculum(self.task_set, self.seed)
        elif self.curriculum_type == "learning_progress":
            return LearningProgressCurriculum(self.task_set, self.num_tasks, self.seed)
        else:
            raise ValueError(f"Unknown curriculum type: {self.curriculum_type}")


# Convenience functions


def weighted_task_set(
    items: List[tuple[Union[EnvConfig, TaskSet], float]],
    overrides: Optional[Union[Dict, List[str]]] = None,
    seed: int = 42,
) -> WeightedTaskSet:
    """Create a WeightedTaskSet directly."""
    return WeightedTaskSet(items, overrides, seed)


def bucketed_task_set(
    base_config: EnvConfig,
    buckets: Dict[str, Any],
    overrides: Optional[Union[Dict, List[str]]] = None,
    seed: int = 42,
) -> BucketedTaskSet:
    """Create a BucketedTaskSet directly."""
    return BucketedTaskSet(base_config, buckets, overrides, seed)


def random_curriculum(task_set: TaskSet, seed: int = 42) -> RandomCurriculum:
    """Create a RandomCurriculum directly."""
    return RandomCurriculum(task_set, seed)


def learning_progress_curriculum(
    task_set: TaskSet, num_tasks: int = 10, seed: int = 42
) -> LearningProgressCurriculum:
    """Create a LearningProgressCurriculum directly."""
    return LearningProgressCurriculum(task_set, num_tasks, seed)


# Builder factory functions


def task_set_builder(seed: int = 42) -> TaskSetBuilder:
    """Create a TaskSetBuilder."""
    return TaskSetBuilder(seed)


def bucketed_builder(base_config: EnvConfig, seed: int = 42) -> BucketedTaskSetBuilder:
    """Create a BucketedTaskSetBuilder."""
    return BucketedTaskSetBuilder(base_config, seed)


def curriculum_builder(task_set: TaskSet) -> CurriculumBuilder:
    """Create a CurriculumBuilder."""
    return CurriculumBuilder(task_set)
