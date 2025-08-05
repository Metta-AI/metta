from pydantic import ConfigDict, Field

from metta.common.util.config import Config
from metta.map.scene import Scene
from metta.map.types import MapGrid


class CopyGridParams(Config):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: MapGrid = Field(exclude=True)  # full outer grid


class CopyGrid(Scene[CopyGridParams]):
    """
    This is a helper scene that allows us to use the preexisting grid as a scene.

    It's main purpose is for MapGen's `instance_map` parameter.
    """

    def render(self):
        if self.width < self.params.grid.shape[1] or self.height < self.params.grid.shape[0]:
            # Shouldn't happen if MapGen is implemented correctly.
            raise ValueError("The area is too small to copy the given grid into it")

        self.grid[:] = "wall"

        # Calculate center position for placing the grid
        source_height, source_width = self.params.grid.shape
        start_row = (self.height - source_height) // 2
        end_row = start_row + source_height
        start_col = (self.width - source_width) // 2
        end_col = start_col + source_width

        # Place the grid at the center
        self.grid[start_row:end_row, start_col:end_col] = self.params.grid
