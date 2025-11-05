"""Evaluation missions and utilities for scripted agent testing."""

from cogames.cogs_vs_clips.evals.difficulty_variants import (
    CANONICAL_DIFFICULTY_ORDER,
    DIFFICULTY_LEVELS,
    apply_difficulty,
)
from cogames.cogs_vs_clips.evals.eval_missions import (
    EVAL_MISSIONS,
)

__all__ = [
    # Difficulty variants
    "DIFFICULTY_LEVELS",
    "CANONICAL_DIFFICULTY_ORDER",
    "apply_difficulty",
    # Utilities / registry
    "EVAL_MISSIONS",
]
