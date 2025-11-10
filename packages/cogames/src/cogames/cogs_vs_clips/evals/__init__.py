"""Evaluation missions and difficulty variants for CoGs vs Clips."""

from cogames.cogs_vs_clips.evals.difficulty_variants import (
    DIFFICULTY_VARIANTS,
    DifficultyLevel,
    get_difficulty,
)
from cogames.cogs_vs_clips.evals.eval_missions import (
    EVAL_MISSIONS,
    MODERATE_SUCCESS_MISSIONS,
    SUCCESSFUL_DIFFICULTIES,
    SUCCESSFUL_MISSIONS,
)

__all__ = [
    "DIFFICULTY_VARIANTS",
    "DifficultyLevel",
    "EVAL_MISSIONS",
    "MODERATE_SUCCESS_MISSIONS",
    "SUCCESSFUL_DIFFICULTIES",
    "SUCCESSFUL_MISSIONS",
    "get_difficulty",
]
