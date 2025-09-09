import numpy as np
from pydantic import ConfigDict, Field

from metta.mettagrid.config import Config
from metta.mettagrid.mapgen.scene import Scene
from metta.mettagrid.mapgen.types import MapGrid
from metta.mettagrid.object_types import ObjectTypes


class CopyGridParams(Config):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: MapGrid = Field(exclude=True)  # full outer grid


class CopyGrid(Scene[CopyGridParams]):
    """
    This is a helper scene that allows us to use the preexisting grid as a scene.

    It's main purpose is for MapGen's `instance_map` parameter.

    MIGRATION NOTE: This scene now supports both legacy string-based grids and new int-based grids.
    The implementation automatically detects the grid format and uses appropriate operations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Detect grid format for migration compatibility
        self._grid_is_int = self.grid.dtype == np.uint8
        self._wall_value = ObjectTypes.WALL if self._grid_is_int else "wall"

    def render(self):
        if self.width < self.params.grid.shape[1] or self.height < self.params.grid.shape[0]:
            # Shouldn't happen if MapGen is implemented correctly.
            raise ValueError("The area is too small to copy the given grid into it")

        self.grid[:] = self._wall_value

        # Calculate center position for placing the grid
        source_height, source_width = self.params.grid.shape
        start_row = (self.height - source_height) // 2
        end_row = start_row + source_height
        start_col = (self.width - source_width) // 2
        end_col = start_col + source_width

        # Place the grid at the center
        self.grid[start_row:end_row, start_col:end_col] = self.params.grid
