"""Cogworks curriculum package."""

from .curriculum import Curriculum, CurriculumConfig, CurriculumTask
from .task import Task
from .task_generator import (
    BucketedTaskGenerator,
    BucketedTaskGeneratorConfig,
    SingleTaskGenerator,
    SingleTaskGeneratorConfig,
    TaskGenerator,
    TaskGeneratorConfig,
    TaskGeneratorSet,
    TaskGeneratorSetConfig,
    ValueRange,
)

__all__ = [
    "Curriculum",
    "CurriculumConfig",
    "CurriculumTask",
    "Task",
    "TaskGenerator",
    "TaskGeneratorConfig",
    "SingleTaskGenerator",
    "SingleTaskGeneratorConfig",
    "TaskGeneratorSet",
    "TaskGeneratorSetConfig",
    "BucketedTaskGenerator",
    "BucketedTaskGeneratorConfig",
    "ValueRange",
]
