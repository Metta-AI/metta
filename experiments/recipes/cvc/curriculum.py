"""Curriculum-focused CoGs vs Clips training entrypoints."""

from experiments.recipes.cogs_v_clips import (
    make_curriculum,
    make_training_env,
    play,
    train,
)

__all__ = ["train", "make_curriculum", "make_training_env", "play"]
