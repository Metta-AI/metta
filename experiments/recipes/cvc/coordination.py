"""Training entrypoint focused on multi-agent coordination missions."""

from metta.tools.train import TrainTool

from experiments.recipes.cvc.core import train_coordination as _train_coordination


def train(*, num_cogs: int = 4) -> TrainTool:
    """Train on missions emphasizing coordination challenges."""
    return _train_coordination(num_cogs=num_cogs)


__all__ = ["train"]
