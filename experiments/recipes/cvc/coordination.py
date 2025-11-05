"""Coordination-focused CoGs vs Clips training entrypoint."""

from experiments.recipes.cvc.core import play, train_coordination as train

__all__ = ["train", "play"]
