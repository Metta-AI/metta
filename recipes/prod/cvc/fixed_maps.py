"""Fixed-map CoGs vs Clips prod entry point."""

from recipes.experiment.cogs_v_clips import play
from recipes.experiment.cogs_v_clips import train_fixed_maps as train

__all__ = ["train", "play"]
