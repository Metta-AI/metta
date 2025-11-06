"""Curriculum-focused CoGs vs Clips training entrypoints."""

from __future__ import annotations

import typing

import experiments.recipes.cvc.core
import metta.cogworks.curriculum.curriculum
import metta.sim.simulation_config
import metta.tools.play
import metta.tools.train
import mettagrid.config.mettagrid_config

__all__ = ["train", "make_curriculum", "make_training_env", "play"]


def train(
    *,
    curriculum: typing.Optional[
        metta.cogworks.curriculum.curriculum.CurriculumConfig
    ] = None,
    base_missions: typing.Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
    num_cogs: int = 4,
) -> metta.tools.train.TrainTool:
    """Run the full curriculum training loop."""
    return experiments.recipes.cvc.core.train(
        curriculum=curriculum,
        base_missions=base_missions,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        num_cogs=num_cogs,
    )


def make_curriculum(
    *,
    num_cogs: int = 4,
    base_missions: typing.Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
) -> metta.cogworks.curriculum.curriculum.CurriculumConfig:
    """Construct the default curriculum configuration."""
    return experiments.recipes.cvc.core.make_curriculum(
        num_cogs=num_cogs,
        base_missions=base_missions,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    )


def make_training_env(
    *,
    mission_name: str = "extractor_hub_30",
    num_cogs: int = 4,
) -> mettagrid.config.mettagrid_config.MettaGridConfig:
    """Build a single training environment from a mission template."""
    return experiments.recipes.cvc.core.make_training_env(
        num_cogs=num_cogs,
        mission_name=mission_name,
    )


def play(
    *,
    policy_uri: typing.Optional[str] = None,
    mission_name: str = "extractor_hub_30",
    num_cogs: int = 4,
) -> metta.tools.play.PlayTool:
    """Play any mission participating in the curriculum."""
    return experiments.recipes.cvc.core.play(
        policy_uri=policy_uri,
        mission_name=mission_name,
        num_cogs=num_cogs,
    )
