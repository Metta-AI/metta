"""Utility functions for mission configuration."""

from __future__ import annotations

from pathlib import Path

from mettagrid.map_builder.map_builder import MapBuilderConfig


def get_map(map_name: str) -> MapBuilderConfig:
    """Load a map by name from the maps directory."""
    maps_dir = Path(__file__).parent.parent / "maps"
    map_path = maps_dir / map_name
    return MapBuilderConfig.from_uri(str(map_path))
