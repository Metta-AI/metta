"""Public curriculum API surface preserved for legacy imports."""

import importlib

import metta.cogworks.curriculum.curriculum
import metta.cogworks.curriculum.curriculum_env
import metta.cogworks.curriculum.learning_progress_algorithm
import metta.cogworks.curriculum.stats
import metta.cogworks.curriculum.task_generator
import metta.cogworks.curriculum.task_tracker
import mettagrid.config.mettagrid_config

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


def single_task(
    mg_config: mettagrid.config.mettagrid_config.MettaGridConfig,
) -> metta.cogworks.curriculum.task_generator.SingleTaskGenerator.Config:
    return metta.cogworks.curriculum.task_generator.SingleTaskGenerator.Config(env=mg_config.model_copy(deep=True))


def bucketed(
    mg_config: mettagrid.config.mettagrid_config.MettaGridConfig,
) -> metta.cogworks.curriculum.task_generator.BucketedTaskGenerator.Config:
    return metta.cogworks.curriculum.task_generator.BucketedTaskGenerator.Config.from_mg(
        mg_config.model_copy(deep=True)
    )


def merge(
    task_generator_configs: list[metta.cogworks.curriculum.task_generator.AnyTaskGeneratorConfig],
) -> metta.cogworks.curriculum.task_generator.TaskGeneratorSet.Config:
    return metta.cogworks.curriculum.task_generator.TaskGeneratorSet.Config(
        task_generators=task_generator_configs, weights=[1.0] * len(task_generator_configs)
    )


def env_curriculum(
    mg_config: mettagrid.config.mettagrid_config.MettaGridConfig,
) -> metta.cogworks.curriculum.curriculum.CurriculumConfig:
    return metta.cogworks.curriculum.curriculum.CurriculumConfig(
        task_generator=metta.cogworks.curriculum.task_generator.SingleTaskGenerator.Config(env=mg_config)
    )


_SUBMODULE_ALIASES = {
    "curriculum": "metta.cogworks.curriculum.curriculum",
    "curriculum_env": "metta.cogworks.curriculum.curriculum_env",
    "learning_progress_algorithm": "metta.cogworks.curriculum.learning_progress_algorithm",
    "stats": "metta.cogworks.curriculum.stats",
    "task_generator": "metta.cogworks.curriculum.task_generator",
    "task_tracker": "metta.cogworks.curriculum.task_tracker",
}

_CLASS_EXPORTS = {
    "Curriculum": ("metta.cogworks.curriculum.curriculum", "Curriculum"),
    "CurriculumConfig": ("metta.cogworks.curriculum.curriculum", "CurriculumConfig"),
    "CurriculumTask": ("metta.cogworks.curriculum.curriculum", "CurriculumTask"),
    "LearningProgressAlgorithm": (
        "metta.cogworks.curriculum.learning_progress_algorithm",
        "LearningProgressAlgorithm",
    ),
    "LearningProgressConfig": ("metta.cogworks.curriculum.learning_progress_algorithm", "LearningProgressConfig"),
    "StatsLogger": ("metta.cogworks.curriculum.stats", "StatsLogger"),
    "SliceAnalyzer": ("metta.cogworks.curriculum.stats", "SliceAnalyzer"),
    "TaskTracker": ("metta.cogworks.curriculum.task_tracker", "TaskTracker"),
    "TaskGenerator": ("metta.cogworks.curriculum.task_generator", "TaskGenerator"),
    "TaskGeneratorConfig": ("metta.cogworks.curriculum.task_generator", "TaskGeneratorConfig"),
    "AnyTaskGeneratorConfig": ("metta.cogworks.curriculum.task_generator", "AnyTaskGeneratorConfig"),
    "SingleTaskGenerator": ("metta.cogworks.curriculum.task_generator", "SingleTaskGenerator"),
    "TaskGeneratorSet": ("metta.cogworks.curriculum.task_generator", "TaskGeneratorSet"),
    "BucketedTaskGenerator": ("metta.cogworks.curriculum.task_generator", "BucketedTaskGenerator"),
    "Span": ("metta.cogworks.curriculum.stats", "Span"),
}


def __getattr__(name: str):
    target = _SUBMODULE_ALIASES.get(name)
    if target is not None:
        module = importlib.import_module(target)
        globals()[name] = module
        return module

    class_target = _CLASS_EXPORTS.get(name)
    if class_target is not None:
        module_path, attr_name = class_target
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module 'metta.cogworks.curriculum' has no attribute '{name}'")
