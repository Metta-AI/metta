"""Backward compatibility shim for metta.cogworks.curriculum.

This module provides deprecated imports from the old location.
All functionality has been moved to the standalone 'agora' package.

Use 'import agora' instead of 'from metta.cogworks.curriculum import ...'.
"""

import warnings

# Emit deprecation warning once when module is imported
warnings.warn(
    "metta.cogworks.curriculum is deprecated and will be removed in a future version. "
    "Use 'import agora' instead. "
    "See packages/agora/README.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from agora for backward compatibility
from agora import (  # noqa: E402
    BasicLPScorer,
    BidirectionalLPScorer,
    BucketedTaskGenerator,
    Curriculum,
    CurriculumAlgorithm,
    CurriculumAlgorithmConfig,
    CurriculumConfig,
    CurriculumTask,
    DiscreteRandomConfig,
    DiscreteRandomCurriculum,
    LearningProgressAlgorithm,
    LearningProgressConfig,
    LocalMemoryBackend,
    LPScorer,
    SharedMemoryBackend,
    SingleTaskGenerator,
    SliceAnalyzer,
    Span,
    StatsLogger,
    TaskGenerator,
    TaskGeneratorConfig,
    TaskGeneratorSet,
    TaskMemoryBackend,
    TaskTracker,
)

# Optional PufferEnv wrapper
try:
    from agora.wrappers import CurriculumEnv  # noqa: E402
except ImportError:
    # pufferlib not available - define a placeholder
    CurriculumEnv = None  # type: ignore[misc, assignment]

# MettaGrid-specific imports (still needed for helper functions)
from mettagrid.config.mettagrid_config import MettaGridConfig  # noqa: E402

# Type alias for backward compatibility
AnyTaskGeneratorConfig = TaskGeneratorConfig  # type: ignore[misc]

__all__ = [
    "Curriculum",
    "CurriculumConfig",
    "CurriculumTask",
    "CurriculumAlgorithm",
    "CurriculumAlgorithmConfig",
    "DiscreteRandomConfig",
    "DiscreteRandomCurriculum",
    "LearningProgressAlgorithm",
    "LearningProgressConfig",
    "StatsLogger",
    "SliceAnalyzer",
    "TaskTracker",
    "TaskMemoryBackend",
    "LocalMemoryBackend",
    "SharedMemoryBackend",
    "LPScorer",
    "BasicLPScorer",
    "BidirectionalLPScorer",
    "TaskGenerator",
    "TaskGeneratorConfig",
    "AnyTaskGeneratorConfig",
    "SingleTaskGenerator",
    "TaskGeneratorSet",
    "BucketedTaskGenerator",
    "Span",
    "CurriculumEnv",
    # Helper functions
    "bucketed",
    "single_task",
    "merge",
    "env_curriculum",
]


def single_task(mg_config: MettaGridConfig) -> SingleTaskGenerator.Config:  # type: ignore[name-defined]
    """Create a `SingleTaskGenerator.Config` from a `MettaGridConfig`.

    DEPRECATED: Use agora.SingleTaskGenerator.Config(env=mg_config) directly.
    """
    return SingleTaskGenerator.Config(env=mg_config.model_copy(deep=True))


def bucketed(mg_config: MettaGridConfig) -> BucketedTaskGenerator.Config:  # type: ignore[name-defined]
    """Create a `BucketedTaskGenerator.Config` from a `MettaGridConfig`.

    DEPRECATED: Use agora.BucketedTaskGenerator.Config.from_base() directly.
    """
    return BucketedTaskGenerator.Config.from_base(mg_config.model_copy(deep=True))


def merge(task_generator_configs: list[TaskGeneratorConfig]) -> TaskGeneratorSet.Config:  # type: ignore[name-defined]
    """Merge configs into a `TaskGeneratorSet.Config`.

    DEPRECATED: Use agora.TaskGeneratorSet.Config() directly.
    """
    return TaskGeneratorSet.Config(task_generators=task_generator_configs, weights=[1.0] * len(task_generator_configs))


def env_curriculum(mg_config: MettaGridConfig) -> CurriculumConfig:
    """Create a curriculum configuration from an MettaGridConfig.

    DEPRECATED: Use agora.CurriculumConfig() directly.
    """
    return CurriculumConfig(task_generator=SingleTaskGenerator.Config(env=mg_config))
