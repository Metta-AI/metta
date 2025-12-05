"""Variant shuffler for generating mission combinations.

Creates random combinations of variants, filters by target difficulty,
and provides tools for evaluation against agents.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cogames.cogs_vs_clips.difficulty_estimator import DifficultyReport, estimate_difficulty
from cogames.cogs_vs_clips.mission import Mission
from cogames.cogs_vs_clips.variants import (
    ChestHeartTuneVariant,
    CompassVariant,
    DarkSideVariant,
    EnergizedVariant,
    HeartChorusVariant,
    InventoryHeartTuneVariant,
    LonelyHeartVariant,
    MinedOutVariant,
    MissionVariant,
    PackRatVariant,
    RoughTerrainVariant,
    Small50Variant,
    SolarFlareVariant,
    SuperChargedVariant,
    TinyHeartProtocolsVariant,
)

if TYPE_CHECKING:
    pass


# Variants that can be combined (no hard conflicts)
COMBINABLE_VARIANTS: list[type[MissionVariant]] = [
    # Energy variants
    DarkSideVariant,  # Zero regen
    EnergizedVariant,  # Max energy + full regen
    SuperChargedVariant,  # +2 regen
    RoughTerrainVariant,  # 2x move cost
    SolarFlareVariant,  # Weaker chargers
    # Recipe variants
    LonelyHeartVariant,  # 1 of each resource
    HeartChorusVariant,  # Multi-agent vibes + reward shaping
    TinyHeartProtocolsVariant,  # Low-cost heart protocols
    # Inventory variants
    PackRatVariant,  # 2x cargo/energy/heart capacity
    InventoryHeartTuneVariant,  # Pre-filled agent inventory
    ChestHeartTuneVariant,  # Pre-filled chest
    # Resource variants
    MinedOutVariant,  # Limited extractor uses
    # Map variants
    Small50Variant,  # 50x50 map
    # Utility variants
    CompassVariant,  # Compass toward assembler
]

# Variants that conflict (can't be used together)
CONFLICT_GROUPS = [
    # Energy conflicts: DarkSide (zero regen) conflicts with regen boosters
    {DarkSideVariant, EnergizedVariant},
    {DarkSideVariant, SuperChargedVariant},
    # Energized already maxes regen, SuperCharged is redundant
    {EnergizedVariant, SuperChargedVariant},
    # Recipe conflicts: LonelyHeart already simplifies recipes
    {LonelyHeartVariant, TinyHeartProtocolsVariant},
]


@dataclass
class VariantCombination:
    """A combination of variants with estimated difficulty."""

    variants: list[MissionVariant]
    variant_names: list[str]
    difficulty_report: DifficultyReport | None = None

    @property
    def difficulty(self) -> float:
        if self.difficulty_report:
            return self.difficulty_report.difficulty_score
        return float("inf")

    @property
    def feasible(self) -> bool:
        if self.difficulty_report:
            return self.difficulty_report.feasible
        return False


@dataclass
class ShufflerConfig:
    """Configuration for variant shuffling."""

    min_variants: int = 0
    max_variants: int = 4
    target_difficulty_min: float = 0.0
    target_difficulty_max: float = float("inf")
    require_feasible: bool = True
    seed: int | None = None


@dataclass
class ShufflerResult:
    """Result of variant shuffling."""

    combinations: list[VariantCombination] = field(default_factory=list)
    rejected_infeasible: int = 0
    rejected_too_easy: int = 0
    rejected_too_hard: int = 0
    total_generated: int = 0


def check_conflicts(variants: list[type[MissionVariant]]) -> bool:
    """Check if a set of variant types has any conflicts."""
    variant_set = set(variants)
    for conflict_group in CONFLICT_GROUPS:
        if len(variant_set & conflict_group) > 1:
            return True
    return False


def generate_variant_combinations(
    n: int,
    config: ShufflerConfig | None = None,
) -> list[list[type[MissionVariant]]]:
    """Generate n random variant combinations.

    Args:
        n: Number of combinations to generate
        config: Shuffler configuration

    Returns:
        List of variant type lists (not instantiated)
    """
    if config is None:
        config = ShufflerConfig()

    rng = random.Random(config.seed)
    combinations: list[list[type[MissionVariant]]] = []

    attempts = 0
    max_attempts = n * 10  # Avoid infinite loops

    while len(combinations) < n and attempts < max_attempts:
        attempts += 1

        # Random number of variants
        num_variants = rng.randint(config.min_variants, config.max_variants)

        # Sample variants
        if num_variants == 0:
            variant_types: list[type[MissionVariant]] = []
        else:
            variant_types = rng.sample(COMBINABLE_VARIANTS, min(num_variants, len(COMBINABLE_VARIANTS)))

        # Check for conflicts
        if check_conflicts(variant_types):
            continue

        # Avoid duplicates
        variant_tuple = tuple(sorted(v.__name__ for v in variant_types))
        existing = [tuple(sorted(v.__name__ for v in c)) for c in combinations]
        if variant_tuple in existing:
            continue

        combinations.append(variant_types)

    return combinations


def shuffle_and_filter(
    base_mission: Mission,
    n_target: int,
    config: ShufflerConfig | None = None,
) -> ShufflerResult:
    """Generate and filter variant combinations by difficulty.

    Args:
        base_mission: Base mission to apply variants to
        n_target: Target number of valid combinations
        config: Shuffler configuration

    Returns:
        ShufflerResult with valid combinations and statistics
    """
    if config is None:
        config = ShufflerConfig()

    result = ShufflerResult()

    # Generate more combinations than needed (many will be filtered)
    raw_combinations = generate_variant_combinations(n_target * 5, config)

    for variant_types in raw_combinations:
        result.total_generated += 1

        # Instantiate variants
        variants = [v() for v in variant_types]
        variant_names = [v.name for v in variants]

        # Apply to mission and estimate difficulty
        mission_with_variants = base_mission.with_variants(variants)
        report = estimate_difficulty(mission_with_variants)

        combo = VariantCombination(
            variants=variants,
            variant_names=variant_names,
            difficulty_report=report,
        )

        # Filter by feasibility
        if config.require_feasible and not combo.feasible:
            result.rejected_infeasible += 1
            continue

        # Filter by difficulty range
        if combo.difficulty < config.target_difficulty_min:
            result.rejected_too_easy += 1
            continue
        if combo.difficulty > config.target_difficulty_max:
            result.rejected_too_hard += 1
            continue

        result.combinations.append(combo)

        if len(result.combinations) >= n_target:
            break

    # Sort by difficulty
    result.combinations.sort(key=lambda c: c.difficulty)

    return result


def create_difficulty_spectrum(
    base_mission: Mission,
    n_per_bucket: int = 3,
    buckets: list[tuple[float, float]] | None = None,
    seed: int | None = None,
) -> dict[str, list[VariantCombination]]:
    """Create a spectrum of missions across difficulty levels.

    Args:
        base_mission: Base mission to apply variants to
        n_per_bucket: Number of combinations per difficulty bucket
        buckets: List of (min, max) difficulty ranges
        seed: Random seed

    Returns:
        Dictionary mapping bucket names to combinations
    """
    if buckets is None:
        buckets = [
            (0.0, 0.1),    # Very easy
            (0.1, 0.3),    # Easy
            (0.3, 0.6),    # Medium
            (0.6, 1.0),    # Hard
            (1.0, 5.0),    # Very hard
        ]

    bucket_names = ["very_easy", "easy", "medium", "hard", "very_hard"]
    spectrum: dict[str, list[VariantCombination]] = {}

    for name, (min_d, max_d) in zip(bucket_names, buckets):
        config = ShufflerConfig(
            target_difficulty_min=min_d,
            target_difficulty_max=max_d,
            seed=seed,
        )
        result = shuffle_and_filter(base_mission, n_per_bucket, config)
        spectrum[name] = result.combinations

    return spectrum


def print_combination(combo: VariantCombination) -> None:
    """Pretty print a variant combination."""
    variants_str = ", ".join(combo.variant_names) if combo.variant_names else "(base)"
    feasible_str = "✓" if combo.feasible else "✗"
    print(f"  {feasible_str} D={combo.difficulty:.3f} | {variants_str}")
    if combo.difficulty_report:
        r = combo.difficulty_report
        print(f"      Steps/♥: {r.steady_state_steps} | Max ♥: {r.max_hearts} | P: {r.success_probability:.0%}")

