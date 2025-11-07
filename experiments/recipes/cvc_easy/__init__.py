"""Easy preset wrappers for CoGs vs Clips recipes (back-compat)."""

from experiments.recipes.cvc.presets import EASY_VARIANTS
from experiments.recipes.cvc.small_maps import play_easy as play, train_easy as train_small_maps

DEFAULT_VARIANTS = EASY_VARIANTS

__all__ = ["DEFAULT_VARIANTS", "play", "train_small_maps"]
