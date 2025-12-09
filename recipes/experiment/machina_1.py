"""Train only on the Machina v1 open world map."""

from __future__ import annotations

from typing import Optional, Sequence

from metta.tools.train import TrainTool
from recipes.experiment.cogs_v_clips import train_single_mission


def train(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    """Entrypoint that locks training to ``machina_1.open_world``."""

    return train_single_mission(
        mission="machina_1.open_world",
        num_cogs=num_cogs,
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
    )


__all__ = ["train"]
