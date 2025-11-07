"""Curriculum-focused CoGs vs Clips training entrypoints."""

from __future__ import annotations

from typing import Optional, Sequence

from experiments.recipes.cogs_v_clips import (
    make_curriculum as base_make_curriculum,
    make_training_env as base_make_training_env,
    play as base_play,
    train as base_train,
)
from experiments.recipes.cvc.presets import (
    EASY_VARIANTS,
    SHAPED_VARIANTS,
    resolve_eval_variants,
    resolve_training_variants,
)
from metta.cogworks.curriculum.curriculum import CurriculumAlgorithmConfig, CurriculumConfig
from mettagrid.config.mettagrid_config import MettaGridConfig
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool


def train(
    *,
    num_cogs: int = 4,
    curriculum: Optional[CurriculumConfig] = None,
    base_missions: Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    return base_train(
        num_cogs=num_cogs,
        curriculum=curriculum,
        base_missions=base_missions,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
    )


def play(
    *,
    policy_uri: Optional[str] = None,
    mission_name: str = "extractor_hub_30",
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
) -> PlayTool:
    return base_play(
        policy_uri=policy_uri,
        mission_name=mission_name,
        num_cogs=num_cogs,
        variants=variants,
    )


def train_easy(
    *,
    num_cogs: int = 4,
    curriculum: Optional[CurriculumConfig] = None,
    base_missions: Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    resolved_variants = resolve_training_variants(EASY_VARIANTS, variants)
    resolved_eval_variants = resolve_eval_variants(eval_variants)
    return base_train(
        num_cogs=num_cogs,
        curriculum=curriculum,
        base_missions=base_missions,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=resolved_variants,
        eval_variants=resolved_eval_variants,
        eval_difficulty=eval_difficulty,
    )


def train_shaped(
    *,
    num_cogs: int = 4,
    curriculum: Optional[CurriculumConfig] = None,
    base_missions: Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    resolved_variants = resolve_training_variants(SHAPED_VARIANTS, variants)
    resolved_eval_variants = resolve_eval_variants(eval_variants)
    return base_train(
        num_cogs=num_cogs,
        curriculum=curriculum,
        base_missions=base_missions,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=resolved_variants,
        eval_variants=resolved_eval_variants,
        eval_difficulty=eval_difficulty,
    )


def play_easy(
    *,
    policy_uri: Optional[str] = None,
    mission_name: str = "extractor_hub_30",
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
) -> PlayTool:
    resolved_variants = resolve_training_variants(EASY_VARIANTS, variants)
    return base_play(
        policy_uri=policy_uri,
        mission_name=mission_name,
        num_cogs=num_cogs,
        variants=resolved_variants,
    )


def play_shaped(
    *,
    policy_uri: Optional[str] = None,
    mission_name: str = "extractor_hub_30",
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
) -> PlayTool:
    resolved_variants = resolve_training_variants(SHAPED_VARIANTS, variants)
    return base_play(
        policy_uri=policy_uri,
        mission_name=mission_name,
        num_cogs=num_cogs,
        variants=resolved_variants,
    )


def make_curriculum(
    *,
    num_cogs: int = 4,
    base_missions: Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
    variants: Optional[Sequence[str]] = None,
) -> CurriculumConfig:
    return base_make_curriculum(
        num_cogs=num_cogs,
        base_missions=base_missions,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        algorithm_config=algorithm_config,
        variants=variants,
    )


def make_training_env(
    *,
    num_cogs: int = 4,
    mission_name: str = "extractor_hub_30",
    variants: Optional[Sequence[str]] = None,
) -> MettaGridConfig:
    return base_make_training_env(
        num_cogs=num_cogs,
        mission_name=mission_name,
        variants=variants,
    )


__all__ = [
    "train",
    "train_easy",
    "train_shaped",
    "make_curriculum",
    "make_training_env",
    "play",
    "play_easy",
    "play_shaped",
]
