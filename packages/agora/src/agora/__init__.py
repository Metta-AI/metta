"""Agora: Environment-agnostic curriculum learning for RL.

Agora provides a complete curriculum learning system that works with any RL environment.
Originally designed for MetaGrid, it's now a standalone package that can be used with
any environment configuration that implements the TaskConfig protocol.

Quick Start:
    >>> from agora import CurriculumConfig, SingleTaskGenerator, LearningProgressConfig
    >>> from pydantic import BaseModel
    >>>
    >>> # Define your task configuration
    >>> class MyTaskConfig(BaseModel):
    ...     difficulty: int = 1
    ...     num_enemies: int = 5
    >>>
    >>> # Create a simple single-task curriculum
    >>> task_gen_config = SingleTaskGenerator.Config(env=MyTaskConfig(difficulty=3))
    >>> curriculum_config = CurriculumConfig(task_generator=task_gen_config)
    >>> curriculum = curriculum_config.make()
    >>>
    >>> # Or use learning progress algorithm
    >>> lp_config = LearningProgressConfig(num_active_tasks=100)
    >>> curriculum_config = CurriculumConfig(
    ...     task_generator=task_gen_config,
    ...     algorithm_config=lp_config
    ... )
    >>> curriculum = curriculum_config.make()
    >>> task = curriculum.get_task()

Main Modules:
    - config: TaskConfig protocol and type definitions
    - curriculum: Core curriculum classes (Curriculum, CurriculumTask, CurriculumAlgorithm)
    - generators: Task generators (SingleTaskGenerator, BucketedTaskGenerator, TaskGeneratorSet)
    - algorithms: Curriculum algorithms (LearningProgressAlgorithm)
    - tracking: Task tracking and statistics (TaskTracker, StatsLogger)
"""

from agora.algorithms.learning_progress import LearningProgressAlgorithm, LearningProgressConfig
from agora.algorithms.scorers import BasicLPScorer, BidirectionalLPScorer, LPScorer
from agora.config import TaskConfig, TConfig
from agora.curriculum import (
    Curriculum,
    CurriculumAlgorithm,
    CurriculumAlgorithmConfig,
    CurriculumConfig,
    CurriculumTask,
    DiscreteRandomConfig,
    DiscreteRandomCurriculum,
)
from agora.generators import (
    BucketedTaskGenerator,
    SingleTaskGenerator,
    Span,
    TaskGenerator,
    TaskGeneratorConfig,
    TaskGeneratorSet,
)
from agora.tracking import (
    LocalMemoryBackend,
    SharedMemoryBackend,
    SliceAnalyzer,
    StatsLogger,
    TaskMemoryBackend,
    TaskTracker,
)

__version__ = "0.1.0"

__all__ = [
    # Core config
    "TaskConfig",
    "TConfig",
    # Curriculum
    "Curriculum",
    "CurriculumConfig",
    "CurriculumTask",
    "CurriculumAlgorithm",
    "CurriculumAlgorithmConfig",
    "DiscreteRandomConfig",
    "DiscreteRandomCurriculum",
    # Generators
    "TaskGenerator",
    "TaskGeneratorConfig",
    "SingleTaskGenerator",
    "BucketedTaskGenerator",
    "TaskGeneratorSet",
    "Span",
    # Algorithms
    "LearningProgressAlgorithm",
    "LearningProgressConfig",
    "LPScorer",
    "BasicLPScorer",
    "BidirectionalLPScorer",
    # Tracking
    "TaskTracker",
    "TaskMemoryBackend",
    "LocalMemoryBackend",
    "SharedMemoryBackend",
    "StatsLogger",
    "SliceAnalyzer",
]
