"""Evaluation missions and utilities for scripted agent testing."""

from cogames.cogs_vs_clips.evals.difficulty_variants import (
    DIFFICULTY_VARIANTS,
)
from cogames.cogs_vs_clips.evals.eval_missions import (
    EVAL_MISSIONS,
    MODERATE_SUCCESS_MISSIONS,
    SUCCESSFUL_DIFFICULTIES,
    SUCCESSFUL_MISSIONS,
)

__all__ = [
    # Difficulty variants
    "DIFFICULTY_VARIANTS",
    # Utilities / registry
    "EVAL_MISSIONS",
    "SUCCESSFUL_MISSIONS",
    "MODERATE_SUCCESS_MISSIONS",
    "SUCCESSFUL_DIFFICULTIES",
]
