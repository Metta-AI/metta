"""Curriculum learning system for adaptive task selection and intelligent training.

This module provides a comprehensive curriculum learning framework that automatically
selects training tasks based on agent learning progress, enabling more efficient and
effective training on complex task distributions.

Core Components:
    - Curriculum: Main curriculum manager coordinating task generation and selection
    - LearningProgressAlgorithm: Intelligent task selection based on learning progress
    - TaskTracker: Performance tracking with shared memory support for multi-process training
    - TaskGenerator: Flexible task generation (single, bucketed, or merged task sets)

Key Features:
    - Bidirectional Learning Progress: Tracks fast/slow EMAs to detect learning opportunities
    - Shared Memory Backend: True multi-process training with atomic updates
    - Comprehensive Statistics: Per-task metrics, Gini coefficients, and learning progress tracking
    - Flexible Task Generation: Support for parameterized, bucketed, and merged task distributions

Quick Start:
    # Basic curriculum with learning progress
    config = LearningProgressConfig(
        use_bidirectional=True,
        num_active_tasks=256,
    )

See Also:
    - learning_progress_algorithm.py: Core LP algorithm with bidirectional scoring
    - task_tracker.py: Performance tracking with shared memory support
    - curriculum.py: Main Curriculum class coordinating all components
"""

from mettagrid.config.mettagrid_config import MettaGridConfig

from .curriculum import Curriculum, CurriculumConfig
from .curriculum_base import CurriculumTask
from .curriculum_env import CurriculumEnv
from .learning_progress_algorithm import LearningProgressAlgorithm, LearningProgressConfig
from .stats import StatsLogger
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
