"""Training entrypoint for medium-map CoGs vs Clips missions."""

from metta.tools.train import TrainTool

from experiments.recipes.cvc.core import train_medium_maps as _train_medium_maps


def train(*, num_cogs: int = 4) -> TrainTool:
    """Train on the medium-map subset (50x50 layouts)."""
    return _train_medium_maps(num_cogs=num_cogs)


__all__ = ["train"]
