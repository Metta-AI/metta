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
        ascii_height, ascii_width = self.ascii_grid.shape
        if self.width < ascii_width + params.column or self.height < ascii_height + params.row:
            raise ValueError(
                f"ASCII grid size {ascii_width}x{ascii_height} is too large"
                f" for area size {self.width}x{self.height} at "
                f"({params.column},{params.row})"
            )

        self.grid[
            params.row : params.row + ascii_height,
            params.column : params.column + ascii_width,
        ] = self.ascii_grid

    @classmethod
    def intrinsic_size(cls, params: InlineAsciiParams) -> tuple[int, int]:
        params = cls.validate_params(params)
        _, width, height = char_grid_to_lines(params.data)
        return height + params.row, width + params.column
