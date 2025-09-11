from metta.mettagrid.mettagrid_config import MettaGridConfig

from .curriculum import (
    Curriculum,
    CurriculumConfig,
    CurriculumTask,
    DiscreteRandomConfig,
    ScoreBasedEvictionPolicy,
    TaskPool,
    TaskSample,
)
from .curriculum_env import CurriculumEnv
from .learning_progress_algorithm import EMAScorer, LearningProgressAlgorithm, LearningProgressConfig
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
from .task_tracker import TaskTracker

# Rebuild models after all imports to resolve forward references
CurriculumConfig.model_rebuild()

__all__ = [
    "Curriculum",
    "CurriculumConfig",
    "CurriculumTask",
    "TaskSample",
    "TaskPool",
    "ScoreBasedEvictionPolicy",
    "DiscreteRandomConfig",
    "LearningProgressAlgorithm",
    "LearningProgressConfig",
    "EMAScorer",
    "TaskTracker",
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
    "merge",
    "env_curriculum",
    "CurriculumEnv",
]


def single_task(mg_config: MettaGridConfig) -> SingleTaskGeneratorConfig:
    """Create a SingleTaskGeneratorConfig from an MettaGridConfig."""
    return SingleTaskGeneratorConfig(env=mg_config.model_copy(deep=True))


def bucketed(mg_config: MettaGridConfig) -> BucketedTaskGeneratorConfig:
    """Create a BucketedTaskGeneratorConfig from an MettaGridConfig."""
    return BucketedTaskGeneratorConfig.from_mg(mg_config.model_copy(deep=True))


def multi_task(*task_generators: AnyTaskGeneratorConfig) -> TaskGeneratorSetConfig:
    """Create a TaskGeneratorSetConfig from multiple task generators."""
    return TaskGeneratorSetConfig(task_generators=list(task_generators))


def merge(*configs: CurriculumConfig) -> CurriculumConfig:
    """Merge multiple CurriculumConfigs into a single config."""
    if not configs:
        raise ValueError("At least one config must be provided")

    # For now, take the first config as base and merge task generators
    base_config = configs[0]

    # Collect all task generators
    all_task_generators = [base_config.task_generator]
    for config in configs[1:]:
        all_task_generators.append(config.task_generator)

    # Create merged task generator set
    merged_task_generator = TaskGeneratorSetConfig(task_generators=all_task_generators)

    # Return new config with merged task generator
    return CurriculumConfig(
        task_generator=merged_task_generator,
        algorithm_config=base_config.algorithm_config,
        num_active_tasks=base_config.num_active_tasks,
        min_presentations_for_eviction=base_config.min_presentations_for_eviction,
        enable_detailed_bucket_logging=base_config.enable_detailed_bucket_logging,
        max_memory_tasks=base_config.max_memory_tasks,
        max_bucket_axes=base_config.max_bucket_axes,
    )


def env_curriculum(env_config: MettaGridConfig) -> CurriculumConfig:
    """Create a simple single-task curriculum from an environment config."""
    return CurriculumConfig(
        task_generator=single_task(env_config),
        algorithm_config=DiscreteRandomConfig(),
    )
