from typing import Optional

from metta.mettagrid.mettagrid_config import EnvConfig

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
    "tasks",
    "curriculum",
]


def tasks(env_config: EnvConfig) -> BucketedTaskGeneratorConfig:
    """Create a BucketedTaskGeneratorConfig from an EnvConfig."""
    return BucketedTaskGeneratorConfig.from_env_config(env_config)


def curriculum(task_generator_config: TaskGeneratorConfig, num_tasks: Optional[int] = None) -> CurriculumConfig:
    """Create a random curriculum configuration."""
    cc = CurriculumConfig(task_generator_config=task_generator_config)
    if num_tasks is not None:
        cc.num_active_tasks = num_tasks
    return cc
