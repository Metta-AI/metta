"""Small-map CoGs vs Clips entry points."""

from recipes.experiment.cogs_v_clips import play
from recipes.experiment.cogs_v_clips import train_small_maps as train

__all__ = ["train", "play"]
