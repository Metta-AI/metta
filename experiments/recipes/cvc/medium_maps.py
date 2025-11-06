"""Medium-map CoGs vs Clips training entrypoint."""

from __future__ import annotations

import typing

import experiments.recipes.cvc.core
import metta.tools.play
import metta.tools.train

__all__ = ["train", "play"]


def train(*, num_cogs: int = 4) -> metta.tools.train.TrainTool:
    """Train on the medium-map curriculum."""
    return experiments.recipes.cvc.core.train_medium_maps(num_cogs=num_cogs)


def play(
    *,
    policy_uri: typing.Optional[str] = None,
    mission_name: str = "extractor_hub_30",
    num_cogs: int = 4,
) -> metta.tools.play.PlayTool:
    """Play a representative medium-map mission."""
    return experiments.recipes.cvc.core.play(
        policy_uri=policy_uri,
        mission_name=mission_name,
        num_cogs=num_cogs,
    )
