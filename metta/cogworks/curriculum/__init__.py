from mettagrid.config.mettagrid_config import MettaGridEnvConfig

from .curriculum import Curriculum, CurriculumConfig, CurriculumTask
from .curriculum_env import CurriculumEnv
from .learning_progress_algorithm import LearningProgressAlgorithm, LearningProgressConfig
from .stats import SliceAnalyzer, StatsLogger
from .task_generator import (
    AnyTaskGeneratorConfig,
    BucketedTaskGenerator,
    SingleTaskGenerator,
    Span,
    TaskGenerator,
    TaskGeneratorConfig,
    TaskGeneratorSet,
)
from .task_tracker import TaskTracker

__all__ = [
    "Curriculum",
    "CurriculumConfig",
    "CurriculumTask",
    "LearningProgressAlgorithm",
    "LearningProgressConfig",
    "StatsLogger",
    "SliceAnalyzer",
    "TaskTracker",
    "TaskGenerator",
    "TaskGeneratorConfig",
    "AnyTaskGeneratorConfig",
    "SingleTaskGenerator",
    "TaskGeneratorSet",
    "BucketedTaskGenerator",
    "Span",
    "bucketed",
    "single_task",
    "merge",
    "env_curriculum",
    "CurriculumEnv",
]


def single_task(mg_config: MettaGridEnvConfig) -> SingleTaskGenerator.Config:
    """Create a `SingleTaskGenerator.Config` from a `MettaGridEnvConfig`."""
    return SingleTaskGenerator.Config(env=mg_config.model_copy(deep=True))


def bucketed(mg_config: MettaGridEnvConfig) -> BucketedTaskGenerator.Config:
    """Create a `BucketedTaskGenerator.Config` from a `MettaGridEnvConfig`."""
    return BucketedTaskGenerator.Config.from_mg(mg_config.model_copy(deep=True))


def merge(task_generator_configs: list[AnyTaskGeneratorConfig]) -> TaskGeneratorSet.Config:
    """Merge configs into a `TaskGeneratorSet.Config`."""
    return TaskGeneratorSet.Config(task_generators=task_generator_configs, weights=[1.0] * len(task_generator_configs))


def env_curriculum(mg_config: MettaGridEnvConfig) -> CurriculumConfig:
    """Create a curriculum configuration from an MettaGridEnvConfig."""
    return CurriculumConfig(task_generator=SingleTaskGenerator.Config(env=mg_config))
