"""Large-map CoGs vs Clips training entrypoints."""

from __future__ import annotations

from typing import Optional, Sequence

from experiments.recipes.cogs_v_clips import (
    play as base_play,
    train_large_maps as base_train_large_maps,
)

EASY_PRESET = "cvc_easy"
SHAPED_PRESET = "cvc_shaped"


def _preset_or_overrides(preset: str, overrides: Optional[Sequence[str]]) -> list[str]:
    return list(overrides) if overrides is not None else [preset]


def train(
    *,
    num_cogs: int = 8,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
):
    return base_train_large_maps(
        num_cogs=num_cogs,
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
    )


def play(
    *,
    policy_uri: Optional[str] = None,
    mission_name: str = "extractor_hub_70",
    num_cogs: int = 8,
    variants: Optional[Sequence[str]] = None,
):
    return base_play(
        policy_uri=policy_uri,
        mission_name=mission_name,
        num_cogs=num_cogs,
        variants=variants,
    )


def train_easy(
    *,
    num_cogs: int = 8,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
):
    return base_train_large_maps(
        num_cogs=num_cogs,
        variants=_preset_or_overrides(EASY_PRESET, variants),
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
    )


def train_shaped(
    *,
    num_cogs: int = 8,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
):
    return base_train_large_maps(
        num_cogs=num_cogs,
        variants=_preset_or_overrides(SHAPED_PRESET, variants),
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
    )


def play_easy(
    *,
    policy_uri: Optional[str] = None,
    mission_name: str = "extractor_hub_70",
    num_cogs: int = 8,
    variants: Optional[Sequence[str]] = None,
):
    return base_play(
        policy_uri=policy_uri,
        mission_name=mission_name,
        num_cogs=num_cogs,
        variants=_preset_or_overrides(EASY_PRESET, variants),
    )


def play_shaped(
    *,
    policy_uri: Optional[str] = None,
    mission_name: str = "extractor_hub_70",
    num_cogs: int = 8,
    variants: Optional[Sequence[str]] = None,
):
    return base_play(
        policy_uri=policy_uri,
        mission_name=mission_name,
        num_cogs=num_cogs,
        variants=_preset_or_overrides(SHAPED_PRESET, variants),
    )


__all__ = [
    "train",
    "play",
    "train_easy",
    "train_shaped",
    "play_easy",
    "play_shaped",
]
