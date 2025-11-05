"""Training entrypoint for large-map CoGs vs Clips missions."""

from metta.tools.train import TrainTool

from experiments.recipes.cvc.core import train_large_maps as _train_large_maps


def train(*, num_cogs: int = 8) -> TrainTool:
    """Train on the large-map subset with more agents."""
    return _train_large_maps(num_cogs=num_cogs)


__all__ = ["train"]
