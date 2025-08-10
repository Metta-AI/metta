"""Curriculum learning components for Metta AI."""

from .rl import (
    # Core classes
    Curriculum,
    Task,
    RandomCurriculum,
    LearningProgressCurriculum,
    TaskSet,
    WeightedTaskSet,
    BuckettedTaskSet,
    create_task_set_from_config,
    # Config classes
    TaskSetConfig,
    WeightedTaskSetConfig,
    BuckettedTaskSetConfig,
    WeightedTaskSetItem,
    BucketValue,
    CurriculumConfig,
    RandomCurriculumConfig,
    LearningProgressCurriculumConfig,
    TaskSetConfigUnion,
    CurriculumConfigUnion,
    # Builder classes
    TaskSetBuilder,
    WeightedTaskSetBuilder,
    BuckettedTaskSetBuilder,
    CurriculumBuilder,
    RandomCurriculumBuilder,
    LearningProgressCurriculumBuilder,
)

__all__ = [
    # Core classes
    "Curriculum",
    "Task",
    "RandomCurriculum", 
    "LearningProgressCurriculum",
    "TaskSet",
    "WeightedTaskSet",
    "BuckettedTaskSet",
    "create_task_set_from_config",
    # Config classes
    "TaskSetConfig",
    "WeightedTaskSetConfig",
    "BuckettedTaskSetConfig",
    "WeightedTaskSetItem",
    "BucketValue",
    "CurriculumConfig",
    "RandomCurriculumConfig",
    "LearningProgressCurriculumConfig",
    "TaskSetConfigUnion",
    "CurriculumConfigUnion",
    # Builder classes
    "TaskSetBuilder",
    "WeightedTaskSetBuilder",
    "BuckettedTaskSetBuilder",
    "CurriculumBuilder",
    "RandomCurriculumBuilder",
    "LearningProgressCurriculumBuilder",
]