import numpy as np

from metta.common.util.config import Config
from metta.map.scene import Scene
from metta.map.utils.ascii_grid import char_grid_to_lines
from metta.mettagrid.char_encoder import char_to_grid_object


class InlineAsciiParams(Config):
    data: str
    row: int = 0
    column: int = 0


class InlineAscii(Scene[InlineAsciiParams]):
    def post_init(self):
        params = self.params

        lines, _, _ = char_grid_to_lines(params.data)
        self.ascii_grid = np.array([list(line) for line in lines], dtype="U6")
        self.ascii_grid = np.vectorize(char_to_grid_object)(self.ascii_grid)

    def render(self):
        params = self.params
        if self.width < self.ascii_grid.shape[1] + params.column or self.height < self.ascii_grid.shape[0] + params.row:
            raise ValueError(
                f"Grid size {self.ascii_grid.shape} is too large for scene size {self.width}x{self.height} at "
                f"{params.column},{params.row}"
            )

        grid_height, grid_width = self.ascii_grid.shape
        self.grid[
            params.row : params.row + grid_height,
            params.column : params.column + grid_width,
        ] = self.ascii_grid
