"""Backwards-compatible wrapper for the easy preset recipe."""

from experiments.recipes.cvc.easy import training as easy_training

train = easy_training.train
play = easy_training.play

__all__ = ["train", "play"]
