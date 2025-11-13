"""Single-mission CoGs vs Clips entry points."""

from recipes.experiment.cogs_v_clips import play
from recipes.experiment.cogs_v_clips import train_single_mission as train

__all__ = ["train", "play"]
