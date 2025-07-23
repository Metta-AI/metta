"""Curriculum module for MettaGrid.

This module provides curriculum learning functionality for MettaGrid environments.
"""

from metta.mettagrid.curriculum.curriculum import Curriculum, MettaGridTask
from metta.mettagrid.curriculum.curriculum_algorithm import (
    CurriculumAlgorithm,
    CurriculumAlgorithmHypers,
    DiscreteRandomCurriculum,
    DiscreteRandomHypers,
)
from metta.mettagrid.curriculum.curriculum_builder import (
    curriculum_config_from_path,
    curriculum_from_config_path,
    parameter_grid_task_set,
    single_task,
    task_set,
)
from metta.mettagrid.curriculum.curriculum_config import CurriculumConfig
from metta.mettagrid.curriculum.learning_progress import LearningProgressHypers
from metta.mettagrid.curriculum.prioritize_regressed import PrioritizeRegressedHypers
from metta.mettagrid.curriculum.progressive import ProgressiveHypers, SimpleProgressiveHypers

__all__ = [
    # Core classes
    "Curriculum",
    "CurriculumConfig",
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
    # Curriculum Creation API
    "task_set",
    "parameter_grid_task_set",
    "single_task",
    "curriculum_config_from_path",
    "curriculum_from_config_path",
]
