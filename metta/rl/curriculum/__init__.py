"""New curriculum system for distributed training with shared memory task pool."""

from .client import CurriculumClient
from .curriculum_env import CurriculumEnv
from .generator import (
    BucketedTaskGenerator,
    CompositeTaskGenerator,
    RandomTaskGenerator,
    SampledTaskGenerator,
    TaskGenerator,
    create_task_generator_from_config,
)
from .manager import CurriculumManager, TaskState
from .task import Task

__all__ = [
    # Manager
    "CurriculumManager",
    "TaskState",
    # Client
    "CurriculumClient",
    # Task
    "Task",
    # Generators
    "TaskGenerator",
    "BucketedTaskGenerator",
    "RandomTaskGenerator",
    "SampledTaskGenerator",
    "CompositeTaskGenerator",
    "create_task_generator_from_config",
    # Environment
    "CurriculumEnv",
]
