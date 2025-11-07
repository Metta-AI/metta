"""Reward-shaped preset wrappers for CoGs vs Clips recipes (back-compat)."""

from experiments.recipes.cvc.presets import SHAPED_VARIANTS
from experiments.recipes.cvc.small_maps import play_shaped as play, train_shaped as train_small_maps

DEFAULT_VARIANTS = SHAPED_VARIANTS

__all__ = ["DEFAULT_VARIANTS", "play", "train_small_maps"]
