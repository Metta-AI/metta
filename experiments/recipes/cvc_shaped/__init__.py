"""Reward-shaped preset wrappers for CoGs vs Clips recipes."""

from __future__ import annotations

from typing import Optional, Sequence

from experiments.recipes.cogs_v_clips import play as base_play
from experiments.recipes.cogs_v_clips import train_small_maps as base_train_small_maps

DEFAULT_VARIANTS: tuple[str, ...] = (
    "lonely_heart",
    "heart_chorus",
    "pack_rat",
    "neutral_faced",
    "extractor_base",
)


def _resolve(variants: Optional[Sequence[str]]) -> list[str]:
    return list(variants) if variants is not None else list(DEFAULT_VARIANTS)


def play(*, variants: Optional[Sequence[str]] = None, **kwargs):
    return base_play(variants=_resolve(variants), **kwargs)


def train_small_maps(
    *,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    **kwargs,
):
    return base_train_small_maps(
        variants=_resolve(variants),
        eval_variants=_resolve(eval_variants),
        **kwargs,
    )

__all__ = ["DEFAULT_VARIANTS", "play", "train_small_maps"]
