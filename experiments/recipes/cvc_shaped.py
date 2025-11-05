"""CoGs vs Clips recipe with reward-shaping oriented defaults."""

from __future__ import annotations

from experiments.recipes.cogs_v_clips_presets import build_cvc_preset

DEFAULT_SHAPED_VARIANTS: tuple[str, ...] = (
    "lonely_heart",
    "heart_chorus",
    "pack_rat",
    "extractor_base",
    "neutral_faced",
)

_PRESET = build_cvc_preset(DEFAULT_SHAPED_VARIANTS)

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
    "DEFAULT_SHAPED_VARIANTS",
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
