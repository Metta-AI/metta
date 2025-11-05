"""Curriculum-focused CoGs vs Clips training entrypoints."""

from functools import partial

from experiments.recipes.cvc.core import (
    make_curriculum,
    make_training_env,
    play as _play,
    train,
)

play = partial(_play, mission_name="extractor_hub_30", num_cogs=4)
play.__doc__ = "Play the default curriculum mission (extractor_hub_30 with 4 cogs)."

__all__ = ["train", "make_curriculum", "make_training_env", "play"]
