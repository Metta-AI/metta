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
    - Dual-Pool Architecture: Separate exploration and exploitation pools with adaptive sampling
    - Shared Memory Backend: True multi-process training with atomic updates
    - Comprehensive Statistics: Per-task metrics, Gini coefficients, pool statistics
    - Flexible Task Generation: Support for parameterized, bucketed, and merged task distributions

Curriculum Modes:
    1. Single-Pool (default): Traditional curriculum with one unified task pool
    2. Dual-Pool: Exploration-exploitation balance with separate pools and adaptive EER
       - Bootstrap phase: 100% exploration until exploit pool fills
       - Steady-state: Adaptive sampling based on promotion success rate

Quick Start:
    # Single-pool curriculum
    config = LearningProgressConfig(
        use_bidirectional=True,
        num_active_tasks=256,
    )

    # Dual-pool curriculum with exploration-exploitation
    config = LearningProgressConfig.default_dual_pool(
        num_explore_tasks=50,
        num_exploit_tasks=200,
    )

See Also:
    - learning_progress_algorithm.py: Core LP algorithm and dual-pool implementation
    - task_tracker.py: Performance tracking and DualPoolTaskTracker
    - curriculum.py: Main Curriculum class coordinating all components
"""

from mettagrid.config.mettagrid_config import MettaGridConfig

from .curriculum import Curriculum, CurriculumConfig, CurriculumTask
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
