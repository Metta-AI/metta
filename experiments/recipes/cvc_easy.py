"""Variant-friendly CoGs vs Clips recipe with easy defaults."""

from __future__ import annotations

from typing import Optional, Sequence

from experiments.recipes.cogs_v_clips import (
    evaluate as base_evaluate,
    make_curriculum as base_make_curriculum,
    make_eval_suite as base_make_eval_suite,
    make_training_env as base_make_training_env,
    play as base_play,
    play_training_env as base_play_training_env,
    train as base_train,
    train_coordination as base_train_coordination,
    train_large_maps as base_train_large_maps,
    train_medium_maps as base_train_medium_maps,
    train_single_mission as base_train_single_mission,
    train_small_maps as base_train_small_maps,
)
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import MettaGridConfig

DEFAULT_EASY_VARIANTS: tuple[str, ...] = (
    "lonely_heart",
    "pack_rat",
    "neutral_faced",
)


def _resolve_variants(variants: Optional[Sequence[str]]) -> list[str]:
    if variants is not None:
        return list(variants)
    return list(DEFAULT_EASY_VARIANTS)


def make_eval_suite(
    num_cogs: int = 4,
    difficulty: str | None = "standard",
    subset: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
) -> list[SimulationConfig]:
    return base_make_eval_suite(
        num_cogs=num_cogs,
        difficulty=difficulty,
        subset=subset,
        variants=_resolve_variants(variants),
    )


def make_training_env(
    num_cogs: int = 4,
    mission_name: str = "extractor_hub_30",
    variants: Optional[Sequence[str]] = None,
) -> MettaGridConfig:
    return base_make_training_env(
        num_cogs=num_cogs,
        mission_name=mission_name,
        variants=_resolve_variants(variants),
    )


def make_curriculum(
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
        variants=_resolve_variants(variants),
    )


def train(
    num_cogs: int = 4,
    curriculum: Optional[CurriculumConfig] = None,
    base_missions: Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    resolved_train_variants = _resolve_variants(variants)
    resolved_eval_variants = (
        list(eval_variants)
        if eval_variants is not None
        else list(resolved_train_variants)
    )
    return base_train(
        num_cogs=num_cogs,
        curriculum=curriculum,
        base_missions=base_missions,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=resolved_train_variants,
        eval_variants=resolved_eval_variants,
        eval_difficulty=eval_difficulty,
    )


def train_small_maps(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    return base_train_small_maps(
        num_cogs=num_cogs,
        variants=_resolve_variants(variants),
        eval_variants=list(eval_variants) if eval_variants is not None else None,
        eval_difficulty=eval_difficulty,
    )


def train_medium_maps(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    return base_train_medium_maps(
        num_cogs=num_cogs,
        variants=_resolve_variants(variants),
        eval_variants=list(eval_variants) if eval_variants is not None else None,
        eval_difficulty=eval_difficulty,
    )


def train_large_maps(
    num_cogs: int = 8,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    return base_train_large_maps(
        num_cogs=num_cogs,
        variants=_resolve_variants(variants),
        eval_variants=list(eval_variants) if eval_variants is not None else None,
        eval_difficulty=eval_difficulty,
    )


def train_coordination(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    return base_train_coordination(
        num_cogs=num_cogs,
        variants=_resolve_variants(variants),
        eval_variants=list(eval_variants) if eval_variants is not None else None,
        eval_difficulty=eval_difficulty,
    )


def train_single_mission(
    mission_name: str = "extractor_hub_30",
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
) -> TrainTool:
    resolved_train_variants = _resolve_variants(variants)
    resolved_eval_variants = (
        list(eval_variants)
        if eval_variants is not None
        else list(resolved_train_variants)
    )
    return base_train_single_mission(
        mission_name=mission_name,
        num_cogs=num_cogs,
        variants=resolved_train_variants,
        eval_variants=resolved_eval_variants,
        eval_difficulty=eval_difficulty,
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
    num_cogs: int = 4,
    difficulty: str | None = "standard",
    subset: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
) -> EvaluateTool:
    return base_evaluate(
        policy_uris=policy_uris,
        num_cogs=num_cogs,
        difficulty=difficulty,
        subset=subset,
        variants=_resolve_variants(variants),
    )


def play(
    policy_uri: Optional[str] = None,
    mission_name: str = "extractor_hub_30",
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
) -> PlayTool:
    return base_play(
        policy_uri=policy_uri,
        mission_name=mission_name,
        num_cogs=num_cogs,
        variants=_resolve_variants(variants),
    )


def play_training_env(
    policy_uri: Optional[str] = None,
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
) -> PlayTool:
    return base_play_training_env(
        policy_uri=policy_uri,
        num_cogs=num_cogs,
        variants=_resolve_variants(variants),
    )


__all__ = [
    "DEFAULT_EASY_VARIANTS",
    "make_eval_suite",
    "make_training_env",
    "make_curriculum",
    "train",
    "train_small_maps",
    "train_medium_maps",
    "train_large_maps",
    "train_coordination",
    "train_single_mission",
    "evaluate",
    "play",
    "play_training_env",
]
