"""Utility functions for mission configuration."""

from __future__ import annotations

from pathlib import Path

from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.mapgen.mapgen import MapGen


def get_map(map_name: str, fixed_spawn_order: bool = False) -> MapBuilderConfig:
    """Load a map by name from the maps directory."""
    maps_dir = Path(__file__).parent.parent / "maps"
    map_path = maps_dir / map_name
    return MapGen.Config(
        instance=MapBuilderConfig.from_uri(str(map_path)),
        fixed_spawn_order=fixed_spawn_order,
        instance_border_width=0,  # Don't add border - maps already have borders built in
    )
