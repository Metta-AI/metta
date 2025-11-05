"""Cogs vs Clips training recipes."""

from experiments.recipes.cogs_v_clips.methods import (
    evaluate,
    make_curriculum,
    make_eval_suite,
    make_training_env,
    play,
    play_training_env,
    train,
    train_coordination,
    train_large_maps,
    train_medium_maps,
    train_single_mission,
    train_small_maps,
)

__all__ = [
    "evaluate",
    "make_curriculum",
    "make_eval_suite",
    "make_training_env",
    "play",
    "play_training_env",
    "train",
    "train_coordination",
    "train_large_maps",
    "train_medium_maps",
    "train_single_mission",
    "train_small_maps",
]
