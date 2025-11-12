"""CVC variant combinations for systematic curriculum exploration.

This module defines standard variant combinations for CoGs vs Clips training.
Each variant modifies agent behavior or reward structure:
  - lonely_heart: Solo-focused behavior
  - heart_chorus: Cooperative behavior
  - pack_rat: Resource hoarding
  - neutral_faced: Baseline/neutral behavior
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

# The 4 core variants we want to explore
CORE_VARIANTS = ["lonely_heart", "heart_chorus", "pack_rat", "neutral_faced"]


def get_single_variants() -> list[tuple[str, ...]]:
    """Get all single variant combinations."""
    return [(v,) for v in CORE_VARIANTS]


def get_variant_pairs() -> list[tuple[str, ...]]:
    """Get all 2-variant combinations (6 total)."""
    return list(combinations(CORE_VARIANTS, 2))


def get_variant_triples() -> list[tuple[str, ...]]:
    """Get all 3-variant combinations (4 total)."""
    return list(combinations(CORE_VARIANTS, 3))


def get_all_variants() -> tuple[str, ...]:
    """Get all 4 variants combined."""
    return tuple(CORE_VARIANTS)


def get_all_combinations() -> list[tuple[str, ...]]:
    """Get all non-empty combinations of the 4 core variants (15 total).

    Returns combinations in order of increasing size:
    - 4 singles
    - 6 pairs
    - 4 triples
    - 1 quad (all variants)
    """
    result: list[tuple[str, ...]] = []
    result.extend(get_single_variants())
    result.extend(get_variant_pairs())
    result.extend(get_variant_triples())
    result.append(get_all_variants())
    return result


def format_variant_name(variants: Sequence[str]) -> str:
    """Format variant combination as a readable name.

    Examples:
        ["lonely_heart"] -> "lonely_heart"
        ["lonely_heart", "pack_rat"] -> "lonely_heart_pack_rat"
    """
    return "_".join(variants)


def get_variant_description(variants: Sequence[str]) -> str:
    """Get human-readable description of variant combination."""
    if len(variants) == 1:
        return f"Single variant: {variants[0]}"
    elif len(variants) == len(CORE_VARIANTS):
        return "All 4 core variants"
    else:
        return f"{len(variants)}-variant combination: {', '.join(variants)}"


__all__ = [
    "CORE_VARIANTS",
    "get_single_variants",
    "get_variant_pairs",
    "get_variant_triples",
    "get_all_variants",
    "get_all_combinations",
    "format_variant_name",
    "get_variant_description",
]
