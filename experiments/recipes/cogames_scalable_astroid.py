from __future__ import annotations

from typing import Optional

from cogames.scalable_astroid import ScalableAstroidParams, make_scalable_arena
from mettagrid.config.mettagrid_config import MettaGridConfig


def make_scalable_astroid_config(
    width: int = 50,
    height: int = 50,
    num_agents: int = 4,
    seed: Optional[int] = None,
    extractor_coverage: float | None = None,
    extractor_names: Optional[list[str]] = None,
    extractor_weights: Optional[dict[str, float]] = None,
    extractor_padding: Optional[int] = None,
    extractor_jitter: Optional[int] = None,
    primary_zone_weights: Optional[dict[str, float]] = None,
    secondary_zone_weights: Optional[dict[str, float]] = None,
    tertiary_zone_weights: Optional[dict[str, float]] = None,
    dungeon_zone_weights: Optional[dict[str, float]] = None,
) -> MettaGridConfig:
    """Expose the scalable astroid arena with defaults for Gridworks."""

    return make_scalable_arena(
        width=width,
        height=height,
        num_agents=num_agents,
        seed=seed,
        params=ScalableAstroidParams(
            extractor_coverage=extractor_coverage,
            extractor_names=extractor_names,
            extractor_weights=extractor_weights,
            extractor_padding=extractor_padding,
            extractor_jitter=extractor_jitter,
            primary_zone_weights=primary_zone_weights,
            secondary_zone_weights=secondary_zone_weights,
            tertiary_zone_weights=tertiary_zone_weights,
            dungeon_zone_weights=dungeon_zone_weights,
        ),
    )
