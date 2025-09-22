"""Map builder that applies operations to create maps."""

from typing import List

from pydantic import ConfigDict
from metta.mettagrid.map_builder import GameMap, MapBuilder, MapBuilderConfig
from metta.mettagrid.map_builder.utils import create_grid
from metta.mettagrid.mapgen.ops import Operation, apply_ops


class OpsMapBuilder(MapBuilder):
    """MapBuilder that creates maps from a list of operations.

    This builder provides a pure functional approach to map generation
    where maps are created by applying a sequence of operations.
    """

    class Config(MapBuilderConfig["OpsMapBuilder"]):
        """Configuration for operations-based map building."""
        model_config = ConfigDict(arbitrary_types_allowed=True)

        # List of operations to apply
        ops: List[Operation]

        # Map dimensions (including border)
        width: int
        height: int

        # Initial fill material before applying operations
        initial_fill: str = "wall"

        # Border width (walls around the edge)
        border_width: int = 1

    def __init__(self, config: Config):
        self.config = config

    def build(self) -> GameMap:
        """Build map from operations."""
        # Create grid with walls
        grid = create_grid(self.config.height, self.config.width)
        grid[:] = "wall"

        # Apply operations to inner area if there's a border
        if self.config.border_width > 0:
            bw = self.config.border_width
            inner_grid = grid[bw:-bw, bw:-bw]
            apply_ops(inner_grid, self.config.ops, initial_fill=self.config.initial_fill)
        else:
            # Apply to entire grid
            apply_ops(grid, self.config.ops, initial_fill=self.config.initial_fill)

        return GameMap(grid)
