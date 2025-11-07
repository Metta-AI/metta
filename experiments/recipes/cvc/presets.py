"""Shared preset helpers for CoGs vs Clips recipes."""

from __future__ import annotations

from typing import Optional, Sequence

EASY_VARIANTS: tuple[str, ...] = ("lonely_heart", "pack_rat", "neutral_faced")
SHAPED_VARIANTS: tuple[str, ...] = (
    "lonely_heart",
    "heart_chorus",
    "pack_rat",
    "neutral_faced",
    "extractor_base",
)


def resolve_training_variants(
    defaults: Sequence[str],
    overrides: Optional[Sequence[str]],
) -> list[str]:
    """Use overrides when provided, otherwise fall back to preset defaults."""

    return list(overrides) if overrides is not None else list(defaults)


def resolve_eval_variants(
    overrides: Optional[Sequence[str]],
) -> Optional[list[str]]:
    """Match training variants by default; only override when explicitly provided."""

    return list(overrides) if overrides is not None else None


__all__ = [
    "EASY_VARIANTS",
    "SHAPED_VARIANTS",
    "resolve_training_variants",
    "resolve_eval_variants",
]
