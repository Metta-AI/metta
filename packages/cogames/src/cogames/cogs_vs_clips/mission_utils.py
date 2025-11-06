"""Utility functions for mission configuration."""

import pathlib

import mettagrid.map_builder.map_builder
import mettagrid.mapgen.mapgen


def get_map(map_name: str, fixed_spawn_order: bool = False) -> mettagrid.map_builder.map_builder.MapBuilderConfig:
    """Load a map by name from the maps directory."""
    maps_dir = pathlib.Path(__file__).parent.parent / "maps"
    map_path = maps_dir / map_name
    return mettagrid.mapgen.mapgen.MapGen.Config(
        instance=mettagrid.map_builder.map_builder.MapBuilderConfig.from_uri(str(map_path)),
        fixed_spawn_order=fixed_spawn_order,
        instance_border_width=0,  # Don't add border - maps already have borders built in
    )
