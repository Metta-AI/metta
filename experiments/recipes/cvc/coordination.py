"""Coordination-focused CoGs vs Clips training entrypoint."""

from __future__ import annotations

import typing

import experiments.recipes.cvc.core
import metta.tools.play
import metta.tools.train

__all__ = ["train", "play"]


def train(*, num_cogs: int = 4) -> metta.tools.train.TrainTool:
    """Train specifically for multi-agent coordination scenarios."""
    return experiments.recipes.cvc.core.train_coordination(num_cogs=num_cogs)


def play(
    *,
    policy_uri: typing.Optional[str] = None,
    mission_name: str = "go_together",
    num_cogs: int = 4,
) -> metta.tools.play.PlayTool:
    """Play a coordination-heavy mission."""
    return experiments.recipes.cvc.core.play(
        policy_uri=policy_uri,
        mission_name=mission_name,
        num_cogs=num_cogs,
    )
