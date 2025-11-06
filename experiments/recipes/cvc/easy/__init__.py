"""Easy preset CoGs vs Clips helpers."""

from __future__ import annotations

from ._preset import DEFAULT_EASY_VARIANTS, PRESET as _PRESET

make_eval_suite = _PRESET.make_eval_suite
make_training_env = _PRESET.make_training_env
make_curriculum = _PRESET.make_curriculum
train = _PRESET.train
train_small_maps = _PRESET.train_small_maps
train_medium_maps = _PRESET.train_medium_maps
train_large_maps = _PRESET.train_large_maps
train_coordination = _PRESET.train_coordination
train_single_mission = _PRESET.train_single_mission
evaluate = _PRESET.evaluate
play = _PRESET.play
play_training_env = _PRESET.play_training_env

__all__ = [
    "DEFAULT_EASY_VARIANTS",
    "make_eval_suite",
    "make_training_env",
    "make_curriculum",
    "train",
    "train_small_maps",
    "train_medium_maps",
    "train_large_maps",
    "train_coordination",
    "train_single_mission",
    "evaluate",
    "play",
    "play_training_env",
]
