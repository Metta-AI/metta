from metta.mettagrid.mettagrid_config import EnvConfig

from .curriculum import Curriculum, CurriculumConfig, CurriculumTask
from .curriculum_env import CurriculumEnv
from .task import Task
from .task_generator import (
    AnyTaskGeneratorConfig,
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
    "AnyTaskGeneratorConfig",
    "SingleTaskGenerator",
    "SingleTaskGeneratorConfig",
    "TaskGeneratorSet",
    "TaskGeneratorSetConfig",
    "BucketedTaskGenerator",
    "BucketedTaskGeneratorConfig",
    "ValueRange",
    "bucketed",
    "multi_task",
    "single_task",
    "curriculum",
    "CurriculumEnv",
]


def single_task(env_config: EnvConfig) -> SingleTaskGeneratorConfig:
    """Create a SingleTaskGeneratorConfig from an EnvConfig."""
    return SingleTaskGeneratorConfig(env=env_config.model_copy(deep=True))


def bucketed(env_config: EnvConfig) -> BucketedTaskGeneratorConfig:
    """Create a BucketedTaskGeneratorConfig from an EnvConfig."""
    return BucketedTaskGeneratorConfig.from_env_config(env_config.model_copy(deep=True))


def multi_task(env_config: EnvConfig) -> TaskGeneratorSetConfig:
    """Create a TaskGeneratorSetConfig from an EnvConfig."""
    return TaskGeneratorSetConfig(
        task_generators=[
            single_task(env_config),
        ],
        weights=[1.0],
    )


def merge(task_generator_configs: list[AnyTaskGeneratorConfig]) -> TaskGeneratorSetConfig:
    """Merge a list of TaskGeneratorConfigs into a TaskGeneratorSetConfig."""
    return TaskGeneratorSetConfig(task_generators=task_generator_configs, weights=[1.0] * len(task_generator_configs))


def env_curriculum(env_config: EnvConfig) -> CurriculumConfig:
    """Create a curriculum configuration from an EnvConfig."""
    return CurriculumConfig(task_generator=SingleTaskGeneratorConfig(env=env_config))
