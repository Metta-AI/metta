from metta.mettagrid.mettagrid_config import EnvConfig

from .curriculum import (
    Curriculum,
    CurriculumAlgorithm,
    CurriculumAlgorithmHypers,
    CurriculumConfig,
    CurriculumTask,
    DiscreteRandomCurriculum,
    DiscreteRandomHypers,
)
from .curriculum_manager import (
    CurriculumManager,
    CurriculumManagerConfig,
)
from .distributed_curriculum import (
    DistributedCurriculumConfig,
    DistributedCurriculumManager,
    SharedMemoryArray,
)
from .learning_progress import (
    LearningProgressCurriculum,
    LearningProgressCurriculumConfig,
    LearningProgressCurriculumTask,
)
from .learning_progress_algorithm import (
    LearningProgressAlgorithm,
    LearningProgressHypers,
)
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
    "CurriculumAlgorithm",
    "CurriculumAlgorithmHypers",
    "DiscreteRandomCurriculum",
    "DiscreteRandomHypers",
    "CurriculumManager",
    "CurriculumManagerConfig",
    "DistributedCurriculumConfig",
    "DistributedCurriculumManager",
    "SharedMemoryArray",
    "LearningProgressCurriculum",
    "LearningProgressCurriculumConfig",
    "LearningProgressCurriculumTask",
    "LearningProgressAlgorithm",
    "LearningProgressHypers",
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
]


def bucketed(env_config: EnvConfig) -> BucketedTaskGeneratorConfig:
    """Create a BucketedTaskGeneratorConfig from an EnvConfig."""
    return BucketedTaskGeneratorConfig.from_env_config(env_config)


def merge(task_generator_configs: list[AnyTaskGeneratorConfig]) -> TaskGeneratorSetConfig:
    """Create a TaskGeneratorSetConfig from a list of TaskGeneratorConfigs."""
    return TaskGeneratorSetConfig(task_generators=task_generator_configs)


def env_curriculum(env_config: EnvConfig) -> CurriculumConfig:
    """Create a CurriculumConfig from an EnvConfig."""
    return bucketed(env_config).to_curriculum()
