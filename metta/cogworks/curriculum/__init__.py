from typing import Optional

from metta.mettagrid.mettagrid_config import EnvConfig

from .curriculum import Curriculum, CurriculumConfig, CurriculumTask
from .curriculum_env import CurriculumEnv
from .task import Task
from .task_generator import (
    BucketedTaskGenerator,
    BucketedTaskGeneratorConfig,
    SingleTaskGenerator,
    SingleTaskGeneratorConfig,
    TaskGenerator,
    TaskGeneratorConfig,
    TaskGeneratorConfigUnion,
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
    "CurriculumEnv",
]


def tasks(env_config: EnvConfig) -> BucketedTaskGeneratorConfig:
    """Create a BucketedTaskGeneratorConfig from an EnvConfig."""
    return BucketedTaskGeneratorConfig.from_env_config(env_config)


def curriculum(task_generator: TaskGeneratorConfigUnion, num_tasks: Optional[int] = None) -> CurriculumConfig:
    """Create a random curriculum configuration."""
    cc = CurriculumConfig(task_generator=task_generator)
    if num_tasks is not None:
        cc.num_active_tasks = num_tasks
    return cc


def env_curriculum(env_config: EnvConfig) -> CurriculumConfig:
    """Create a curriculum configuration from an EnvConfig."""
    return CurriculumConfig(task_generator=SingleTaskGeneratorConfig(env=env_config))
