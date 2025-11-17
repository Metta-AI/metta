"""Coordination-focused CoGs vs Clips prod entry points."""

from recipes.experiment.cogs_v_clips import play
from recipes.experiment.cogs_v_clips import train_coordination as train

__all__ = ["train", "play"]
