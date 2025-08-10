"""Reinforcement learning curriculum components."""

# Task
from .curriculum import Task

# TaskSet
from .config import (
    TaskSetConfig,
    WeightedTaskSetConfig,
    BuckettedTaskSetConfig,
    WeightedTaskSetItem,
    BucketValue,
    TaskSetConfigUnion,
)
from .task_set import TaskSet, WeightedTaskSet, BuckettedTaskSet, create_task_set_from_config
from .builders import (
    TaskSetBuilder,
    WeightedTaskSetBuilder,
    BuckettedTaskSetBuilder,
)

# Curriculum
from .config import (
    CurriculumConfig,
    RandomCurriculumConfig,
    LearningProgressCurriculumConfig,
    CurriculumConfigUnion,
)
from .curriculum import Curriculum, RandomCurriculum, LearningProgressCurriculum
from .builders import (
    CurriculumBuilder,
    RandomCurriculumBuilder,
    LearningProgressCurriculumBuilder,
)

__all__ = [
    # Task
    "Task",
    # TaskSet
    "TaskSetConfig",
    "WeightedTaskSetConfig",
    "BuckettedTaskSetConfig",
    "WeightedTaskSetItem",
    "BucketValue",
    "TaskSetConfigUnion",
    "TaskSet",
    "WeightedTaskSet",
    "BuckettedTaskSet",
    "create_task_set_from_config",
    "TaskSetBuilder",
    "WeightedTaskSetBuilder",
    "BuckettedTaskSetBuilder",
    # Curriculum
    "CurriculumConfig",
    "RandomCurriculumConfig",
    "LearningProgressCurriculumConfig",
    "CurriculumConfigUnion",
    "Curriculum",
    "RandomCurriculum",
    "LearningProgressCurriculum",
    "CurriculumBuilder",
    "RandomCurriculumBuilder",
    "LearningProgressCurriculumBuilder",
]