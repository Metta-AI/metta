"""Evaluation missions and utilities for scripted agent testing."""

import cogames.cogs_vs_clips.evals.difficulty_variants as difficulty_variants
import cogames.cogs_vs_clips.evals.eval_missions as eval_missions

DIFFICULTY_VARIANTS = difficulty_variants.DIFFICULTY_VARIANTS
EVAL_MISSIONS = eval_missions.EVAL_MISSIONS
SUCCESSFUL_MISSIONS = eval_missions.SUCCESSFUL_MISSIONS
MODERATE_SUCCESS_MISSIONS = eval_missions.MODERATE_SUCCESS_MISSIONS
SUCCESSFUL_DIFFICULTIES = eval_missions.SUCCESSFUL_DIFFICULTIES

__all__ = [
    # Difficulty variants
    "DIFFICULTY_VARIANTS",
    # Utilities / registry
    "EVAL_MISSIONS",
    "SUCCESSFUL_MISSIONS",
    "MODERATE_SUCCESS_MISSIONS",
    "SUCCESSFUL_DIFFICULTIES",
]
