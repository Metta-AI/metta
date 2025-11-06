"""Public curriculum API surface preserved for legacy imports."""

import importlib

from mettagrid.config.mettagrid_config import MettaGridConfig

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
    "curriculum",
    "curriculum_env",
    "learning_progress_algorithm",
    "stats",
    "task_generator",
    "task_tracker",
]


def single_task(mg_config: MettaGridConfig) -> SingleTaskGenerator.Config:
    return SingleTaskGenerator.Config(env=mg_config.model_copy(deep=True))


def bucketed(mg_config: MettaGridConfig) -> BucketedTaskGenerator.Config:
    return BucketedTaskGenerator.Config.from_mg(mg_config.model_copy(deep=True))


def merge(task_generator_configs: list[AnyTaskGeneratorConfig]) -> TaskGeneratorSet.Config:
    return TaskGeneratorSet.Config(task_generators=task_generator_configs, weights=[1.0] * len(task_generator_configs))


def env_curriculum(mg_config: MettaGridConfig) -> CurriculumConfig:
    return CurriculumConfig(task_generator=SingleTaskGenerator.Config(env=mg_config))


_SUBMODULE_ALIASES = {
    "curriculum": "metta.cogworks.curriculum.curriculum",
    "curriculum_env": "metta.cogworks.curriculum.curriculum_env",
    "learning_progress_algorithm": "metta.cogworks.curriculum.learning_progress_algorithm",
    "stats": "metta.cogworks.curriculum.stats",
    "task_generator": "metta.cogworks.curriculum.task_generator",
    "task_tracker": "metta.cogworks.curriculum.task_tracker",
}


def __getattr__(name: str):
    target = _SUBMODULE_ALIASES.get(name)
    if target is None:
        raise AttributeError(f"module 'metta.cogworks.curriculum' has no attribute '{name}'")
    module = importlib.import_module(target)
    globals()[name] = module
    return module
