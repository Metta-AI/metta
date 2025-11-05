"""Large-map CoGs vs Clips training entrypoint."""

from functools import partial

from experiments.recipes.cvc.core import play as _play, train_large_maps as train

play = partial(_play, mission_name="extractor_hub_70", num_cogs=8)
play.__doc__ = "Play a large-map mission (defaults to extractor_hub_70 with 8 cogs)."

__all__ = ["train", "play"]
