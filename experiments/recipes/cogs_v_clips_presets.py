"""Utilities for building CoGs vs Clips recipe presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

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


@dataclass(frozen=True)
class CvcPreset:
    """Collection of callable wrappers bound to default variant presets."""

    default_variants: tuple[str, ...]
    make_eval_suite: Callable[
        [int, str | None, Optional[Sequence[str]], Optional[Sequence[str]]],
        list[SimulationConfig],
    ]
    make_training_env: Callable[[int, str, Optional[Sequence[str]]], MettaGridConfig]
    make_curriculum: Callable[
        [
            int,
            Optional[list[str]],
            bool,
            Optional[CurriculumAlgorithmConfig],
            Optional[Sequence[str]],
        ],
        CurriculumConfig,
    ]
    train: Callable[
        [
            int,
            Optional[CurriculumConfig],
            Optional[list[str]],
            bool,
            Optional[Sequence[str]],
            Optional[Sequence[str]],
            str | None,
        ],
        TrainTool,
    ]
    train_small_maps: Callable[
        [int, Optional[Sequence[str]], Optional[Sequence[str]], str | None], TrainTool
    ]
    train_medium_maps: Callable[
        [int, Optional[Sequence[str]], Optional[Sequence[str]], str | None], TrainTool
    ]
    train_large_maps: Callable[
        [int, Optional[Sequence[str]], Optional[Sequence[str]], str | None], TrainTool
    ]
    train_coordination: Callable[
        [int, Optional[Sequence[str]], Optional[Sequence[str]], str | None], TrainTool
    ]
    train_single_mission: Callable[
        [str, int, Optional[Sequence[str]], Optional[Sequence[str]], str | None],
        TrainTool,
    ]
    evaluate: Callable[
        [
            str | Sequence[str] | None,
            int,
            str | None,
            Optional[Sequence[str]],
            Optional[Sequence[str]],
        ],
        EvaluateTool,
    ]
    play: Callable[[Optional[str], str, int, Optional[Sequence[str]]], PlayTool]
    play_training_env: Callable[[Optional[str], int, Optional[Sequence[str]]], PlayTool]


def _resolve_variants(
    default_variants: tuple[str, ...],
    variants: Optional[Sequence[str]],
) -> list[str]:
    if variants is not None:
        return list(variants)
    return list(default_variants)


def build_cvc_preset(default_variants: Sequence[str]) -> CvcPreset:
    """Create a wrapper set that pins default variants for CoGs vs Clips recipes."""

    defaults = tuple(default_variants)

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
            variants=_resolve_variants(defaults, variants),
        )

    def make_training_env(
        num_cogs: int = 4,
        mission_name: str = "extractor_hub_30",
        variants: Optional[Sequence[str]] = None,
    ) -> MettaGridConfig:
        return base_make_training_env(
            num_cogs=num_cogs,
            mission_name=mission_name,
            variants=_resolve_variants(defaults, variants),
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
            variants=_resolve_variants(defaults, variants),
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
        resolved_variants = _resolve_variants(defaults, variants)
        return base_train(
            num_cogs=num_cogs,
            curriculum=curriculum,
            base_missions=base_missions,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
            variants=resolved_variants,
            eval_variants=eval_variants,
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
            variants=_resolve_variants(defaults, variants),
            eval_variants=eval_variants,
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
            variants=_resolve_variants(defaults, variants),
            eval_variants=eval_variants,
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
            variants=_resolve_variants(defaults, variants),
            eval_variants=eval_variants,
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
            variants=_resolve_variants(defaults, variants),
            eval_variants=eval_variants,
            eval_difficulty=eval_difficulty,
        )

    def train_single_mission(
        mission_name: str = "extractor_hub_30",
        num_cogs: int = 4,
        variants: Optional[Sequence[str]] = None,
        eval_variants: Optional[Sequence[str]] = None,
        eval_difficulty: str | None = "standard",
    ) -> TrainTool:
        return base_train_single_mission(
            mission_name=mission_name,
            num_cogs=num_cogs,
            variants=_resolve_variants(defaults, variants),
            eval_variants=eval_variants,
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
            variants=_resolve_variants(defaults, variants),
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
            variants=_resolve_variants(defaults, variants),
        )

    def play_training_env(
        policy_uri: Optional[str] = None,
        num_cogs: int = 4,
        variants: Optional[Sequence[str]] = None,
    ) -> PlayTool:
        return base_play_training_env(
            policy_uri=policy_uri,
            num_cogs=num_cogs,
            variants=_resolve_variants(defaults, variants),
        )

    return CvcPreset(
        default_variants=defaults,
        make_eval_suite=make_eval_suite,
        make_training_env=make_training_env,
        make_curriculum=make_curriculum,
        train=train,
        train_small_maps=train_small_maps,
        train_medium_maps=train_medium_maps,
        train_large_maps=train_large_maps,
        train_coordination=train_coordination,
        train_single_mission=train_single_mission,
        evaluate=evaluate,
        play=play,
        play_training_env=play_training_env,
    )


__all__ = ["CvcPreset", "build_cvc_preset"]
