"""Test support utilities for map building."""

from __future__ import annotations

import numpy as np

from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig


class ObjectNameMapBuilder(MapBuilder):
    """Map builder that uses pre-built object name maps.

    This is primarily used in tests where maps are defined as simple 2D arrays
    of object names (e.g., "wall", "agent.agent", "empty").
    """

    class Config(MapBuilderConfig["ObjectNameMapBuilder"]):
        """Configuration for ObjectNameMapBuilder."""

        map_data: list[list[str]]

    def __init__(self, config: Config):
        self.config = config

    def build(self) -> GameMap:
        return GameMap(grid=np.array(self.config.map_data))
