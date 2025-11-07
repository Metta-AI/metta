"""Easy preset wrappers for CoGs vs Clips recipes."""

from __future__ import annotations

from typing import Optional, Sequence

from experiments.recipes.cogs_v_clips import play as base_play
from experiments.recipes.cogs_v_clips import train_small_maps as base_train_small_maps

DEFAULT_VARIANTS: tuple[str, ...] = (
    "lonely_heart",
    "pack_rat",
    "neutral_faced",
)


def _resolve(variants: Optional[Sequence[str]]) -> list[str]:
    return list(variants) if variants is not None else list(DEFAULT_VARIANTS)


def play(
    *,
    policy_uri: Optional[str] = None,
    mission_name: str = "extractor_hub_30",
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
):
    return base_play(
        policy_uri=policy_uri,
        mission_name=mission_name,
        num_cogs=num_cogs,
        variants=_resolve(variants),
    )


def train_small_maps(
    *,
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
):
    resolved_variants = _resolve(variants)
    resolved_eval_variants = _resolve(eval_variants)
    return base_train_small_maps(
        num_cogs=num_cogs,
        variants=resolved_variants,
        eval_variants=resolved_eval_variants,
        eval_difficulty=eval_difficulty,
    )


__all__ = ["DEFAULT_VARIANTS", "play", "train_small_maps"]
