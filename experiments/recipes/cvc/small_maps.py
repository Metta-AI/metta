"""Small-map CoGs vs Clips training entrypoints."""

from functools import partial

from experiments.recipes.cvc.core import play as _play, train_small_maps as train

play = partial(_play, mission_name="extractor_hub_30", num_cogs=4)
play.__doc__ = "Play a small-map mission (defaults to extractor_hub_30 with 4 cogs)."

__all__ = ["train", "play"]
