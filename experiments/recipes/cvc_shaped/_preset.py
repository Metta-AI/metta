"""Preset configuration for cvc_shaped recipe variants."""

from __future__ import annotations

from experiments.recipes.cogs_v_clips_presets import build_cvc_preset

DEFAULT_SHAPED_VARIANTS: tuple[str, ...] = (
    "lonely_heart",
    "heart_chorus",
    "pack_rat",
    "extractor_base",
    "neutral_faced",
)

PRESET = build_cvc_preset(DEFAULT_SHAPED_VARIANTS)

__all__ = ["DEFAULT_SHAPED_VARIANTS", "PRESET"]
