"""Core curriculum-based training entrypoints for the CVC recipe."""

from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.tools.train import TrainTool

from experiments.recipes.cvc.core import (
    make_curriculum,
    make_training_env,
    train as _train,
)


def train(
    *,
    num_cogs: int = 4,
    curriculum: CurriculumConfig | None = None,
    base_missions: list[str] | None = None,
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    """Train on the full CoGs vs Clips curriculum."""
    return _train(
        num_cogs=num_cogs,
        curriculum=curriculum,
        base_missions=base_missions,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    )


__all__ = ["train", "make_curriculum", "make_training_env"]
