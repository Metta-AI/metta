from __future__ import annotations

from typing import TYPE_CHECKING

from metta.cogworks.curriculum.curriculum import (
    CurriculumConfig,
    CurriculumTask,
)
from metta.cogworks.curriculum.learning_progress_algorithm import (
    LearningProgressAlgorithm,
    LearningProgressConfig,
)
from metta.cogworks.curriculum.stats import SliceAnalyzer, StatsLogger
from metta.cogworks.curriculum.task_generator import (
    AnyTaskGeneratorConfig,
    BucketedTaskGenerator,
    SingleTaskGenerator,
    Span,
    TaskGenerator,
    TaskGeneratorConfig,
    TaskGeneratorSet,
)
from metta.cogworks.curriculum.task_tracker import TaskTracker

if TYPE_CHECKING:
    from metta.cogworks.curriculum.curriculum import Curriculum
    from metta.cogworks.curriculum.curriculum_env import CurriculumEnv

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


def single_task(mg_config: MettaGridConfig) -> SingleTaskGenerator.Config:
    """Create a `SingleTaskGenerator.Config` from a `MettaGridConfig`."""
    return SingleTaskGenerator.Config(env=mg_config.model_copy(deep=True))


def bucketed(mg_config: MettaGridConfig) -> BucketedTaskGenerator.Config:
    """Create a `BucketedTaskGenerator.Config` from a `MettaGridConfig`."""
    return BucketedTaskGenerator.Config.from_mg(mg_config.model_copy(deep=True))


def merge(task_generator_configs: list[AnyTaskGeneratorConfig]) -> TaskGeneratorSet.Config:
    """Merge configs into a `TaskGeneratorSet.Config`."""
    return TaskGeneratorSet.Config(task_generators=task_generator_configs, weights=[1.0] * len(task_generator_configs))


def env_curriculum(mg_config: MettaGridConfig) -> CurriculumConfig:
    """Create a curriculum configuration from an MettaGridConfig."""
    return CurriculumConfig(task_generator=SingleTaskGenerator.Config(env=mg_config))


def __getattr__(name: str):
    """Lazily import CurriculumEnv to avoid loading pufferlib at package import."""
    if name == "Curriculum":
        from metta.cogworks.curriculum.curriculum import CurriculumEnv

        globals()["Curriculum"] = Curriculum
        return Curriculum
    if name == "CurriculumEnv":
        from metta.cogworks.curriculum.curriculum_env import CurriculumEnv

        globals()["CurriculumEnv"] = CurriculumEnv
        return CurriculumEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
