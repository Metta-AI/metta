"""Test support utilities for map building."""

from __future__ import annotations

import numpy as np

from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig


class ObjectNameMapBuilderConfig(MapBuilderConfig["ObjectNameMapBuilder"]):
    """Configuration for ObjectNameMapBuilder."""

    map_data: list[list[str]]


class ObjectNameMapBuilder(MapBuilder[ObjectNameMapBuilderConfig]):
    """Map builder that uses pre-built object name maps.

    This is primarily used in tests where maps are defined as simple 2D arrays
    of object names (e.g., "wall", "agent.agent", "empty").
    """

    def __init__(self, config: ObjectNameMapBuilderConfig):
        super().__init__(config)

    def build(self) -> GameMap:
        return GameMap(grid=np.array(self.config.map_data))
