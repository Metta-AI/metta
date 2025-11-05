"""Single-mission CoGs vs Clips training entrypoint."""

from experiments.recipes.cogs_v_clips import play, train_single_mission as train

__all__ = ["train", "play"]
