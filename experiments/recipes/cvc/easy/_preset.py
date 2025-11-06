"""Preset configuration for easy CoGs vs Clips variants."""

from __future__ import annotations

from experiments.recipes.cogs_v_clips_presets import build_cvc_preset

DEFAULT_EASY_VARIANTS: tuple[str, ...] = (
    "lonely_heart",
    "pack_rat",
    "neutral_faced",
)

PRESET = build_cvc_preset(DEFAULT_EASY_VARIANTS)

__all__ = ["DEFAULT_EASY_VARIANTS", "PRESET"]
