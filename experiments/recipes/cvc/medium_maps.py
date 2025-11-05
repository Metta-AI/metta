"""Medium-map CoGs vs Clips training entrypoint."""

from functools import partial

from experiments.recipes.cvc.core import play as _play, train_medium_maps as train

play = partial(_play, mission_name="extractor_hub_50", num_cogs=4)
play.__doc__ = "Play a medium-map mission (defaults to extractor_hub_50 with 4 cogs)."

__all__ = ["train", "play"]
