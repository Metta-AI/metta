"""Reward-shaped preset wrappers for CoGs vs Clips recipes (back-compat)."""

from experiments.recipes.cvc.small_maps import play_shaped as play, train_shaped as train_small_maps

DEFAULT_VARIANTS = ("cvc_shaped",)

__all__ = ["DEFAULT_VARIANTS", "play", "train_small_maps"]
