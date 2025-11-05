"""Curriculum-focused CoGs vs Clips training entrypoints."""

from experiments.recipes.cvc.core import (
    make_curriculum,
    make_training_env,
    train,
)

__all__ = ["train", "make_curriculum", "make_training_env"]
