"""Curriculum module for MettaGrid.

This module provides curriculum learning functionality for MettaGrid environments.
"""

from metta.mettagrid.curriculum.curriculum_algorithm import (
    CurriculumAlgorithm,
    CurriculumAlgorithmHypers,
    DiscreteRandomCurriculum,
    DiscreteRandomHypers,
)
from metta.mettagrid.curriculum.curriculum_api import (
    parameter_grid_task_set,
    single_task_tree,
    task_set,
)
from metta.mettagrid.curriculum.learning_progress import LearningProgressHypers
from metta.mettagrid.curriculum.prioritize_regressed import PrioritizeRegressedHypers
from metta.mettagrid.curriculum.progressive import ProgressiveHypers, SimpleProgressiveHypers
from metta.mettagrid.curriculum.task_tree import MettaGridTask, TaskTree
from metta.mettagrid.curriculum.task_tree_config import TaskTreeConfig

__all__ = [
    # Core classes
    "TaskTree",
    "MettaGridTask",
    # Algorithm classes
    "CurriculumAlgorithm",
    "CurriculumAlgorithmHypers",
    "DiscreteRandomCurriculum",
    "DiscreteRandomHypers",
    "LearningProgressHypers",
    "PrioritizeRegressedHypers",
    "ProgressiveHypers",
    "SimpleProgressiveHypers",
    # TaskTree Creation API
    "task_set",
    "parameter_grid_task_set",
    "single_task_tree",
    # Config-based creation
    "TaskTreeConfig",
]
