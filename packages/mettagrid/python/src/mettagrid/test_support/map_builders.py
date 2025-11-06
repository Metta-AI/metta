"""Test support utilities for map building."""

import numpy as np

import mettagrid.map_builder.map_builder


class ObjectNameMapBuilderConfig(mettagrid.map_builder.map_builder.MapBuilderConfig["ObjectNameMapBuilder"]):
    """Configuration for ObjectNameMapBuilder."""

    map_data: list[list[str]]


class ObjectNameMapBuilder(mettagrid.map_builder.map_builder.MapBuilder[ObjectNameMapBuilderConfig]):
    """Map builder that uses pre-built object name maps.

    This is primarily used in tests where maps are defined as simple 2D arrays
    of object names (e.g., "wall", "agent.agent", "empty").
    """

    def __init__(self, config: ObjectNameMapBuilderConfig):
        super().__init__(config)

    def build(self) -> mettagrid.map_builder.map_builder.GameMap:
        return mettagrid.map_builder.map_builder.GameMap(grid=np.array(self.config.map_data))
