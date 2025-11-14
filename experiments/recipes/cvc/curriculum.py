"""Curriculum CoGs vs Clips entry points."""

from recipes.experiment.cogs_v_clips import (
    make_curriculum,
    make_training_env,
    play,
    train,
)

__all__ = ["train", "make_curriculum", "make_training_env", "play"]
