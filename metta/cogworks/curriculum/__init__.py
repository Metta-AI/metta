from metta.mettagrid.mettagrid_config import MettaGridConfig

from .curriculum import Curriculum, CurriculumConfig, CurriculumTask
from .curriculum_env import CurriculumEnv
from .task import Task
from .task_generator import (
    AnyTaskGeneratorConfig,
    BucketedTaskGenerator,
    BucketedTaskGeneratorConfig,
    SingleTaskGenerator,
    SingleTaskGeneratorConfig,
    Span,
    TaskGenerator,
    TaskGeneratorConfig,
    TaskGeneratorSet,
    TaskGeneratorSetConfig,
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
    "Span",
    "bucketed",
    "multi_task",
    "single_task",
    "curriculum",
    "CurriculumEnv",
]


def single_task(mg_config: MettaGridConfig) -> SingleTaskGeneratorConfig:
    """Create a SingleTaskGeneratorConfig from an MettaGridConfig."""
    return SingleTaskGeneratorConfig(env=mg_config.model_copy(deep=True))


def bucketed(mg_config: MettaGridConfig) -> BucketedTaskGeneratorConfig:
    """Create a BucketedTaskGeneratorConfig from an MettaGridConfig."""
    return BucketedTaskGeneratorConfig.from_mg(mg_config.model_copy(deep=True))


def multi_task(mg_config: MettaGridConfig) -> TaskGeneratorSetConfig:
    """Create a TaskGeneratorSetConfig from an MettaGridConfig."""
    return TaskGeneratorSetConfig(
        task_generators=[
            single_task(mg_config),
        ],
        weights=[1.0],
    )


def merge(task_generator_configs: list[AnyTaskGeneratorConfig]) -> TaskGeneratorSetConfig:
    """Merge a list of TaskGeneratorConfigs into a TaskGeneratorSetConfig."""
    return TaskGeneratorSetConfig(task_generators=task_generator_configs, weights=[1.0] * len(task_generator_configs))


def env_curriculum(mg_config: MettaGridConfig) -> CurriculumConfig:
    """Create a curriculum configuration from an MettaGridConfig."""
    return CurriculumConfig(task_generator=SingleTaskGeneratorConfig(env=mg_config))
