"""Training entrypoint for small-map CoGs vs Clips missions."""

from metta.tools.train import TrainTool

from experiments.recipes.cvc.core import train_small_maps as _train_small_maps


def train(*, num_cogs: int = 4) -> TrainTool:
    """Train on the small-map subset (30x30 layouts)."""
    return _train_small_maps(num_cogs=num_cogs)


__all__ = ["train"]
