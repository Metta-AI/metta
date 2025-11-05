"""Training entrypoint for debugging on a single mission."""

from metta.tools.train import TrainTool

from experiments.recipes.cvc.core import train_single_mission as _train_single_mission


def train(*, mission_name: str = "extractor_hub_30", num_cogs: int = 4) -> TrainTool:
    """Train on a single mission without a curriculum."""
    return _train_single_mission(mission_name=mission_name, num_cogs=num_cogs)


__all__ = ["train"]
