"""Reinforcement learning curriculum components."""

from .curriculum import Curriculum, Task, RandomCurriculum, LearningProgressCurriculum
from .task_set import TaskSet, WeightedTaskSet, BuckettedTaskSet, create_task_set_from_config
from .config import (
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
)
from .builders import (
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