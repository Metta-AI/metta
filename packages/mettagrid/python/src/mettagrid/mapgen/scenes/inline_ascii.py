import numpy as np

from mettagrid.mapgen.scene import Scene, SceneConfig
from mettagrid.mapgen.utils.ascii_grid import char_grid_to_lines
from mettagrid.util.char_encoder import char_to_grid_object


class InlineAsciiConfig(SceneConfig):
    data: str
    row: int = 0
    column: int = 0


class InlineAscii(Scene[InlineAsciiConfig]):
    def post_init(self):
        config = self.config

        lines, _, _ = char_grid_to_lines(config.data)
        self.ascii_grid = np.array([list(line) for line in lines], dtype="U6")
        self.ascii_grid = np.vectorize(char_to_grid_object)(self.ascii_grid)

    def render(self):
        config = self.config
        ascii_height, ascii_width = self.ascii_grid.shape
        if self.width < ascii_width + config.column or self.height < ascii_height + config.row:
            raise ValueError(
                f"ASCII grid size {ascii_width}x{ascii_height} is too large"
                f" for area size {self.width}x{self.height} at "
                f"({config.column},{config.row})"
            )

        self.grid[
            config.row : config.row + ascii_height,
            config.column : config.column + ascii_width,
        ] = self.ascii_grid

    @classmethod
    def intrinsic_size(cls, config: InlineAsciiConfig) -> tuple[int, int]:
        config = cls.Config.model_validate(config)
        _, width, height = char_grid_to_lines(config.data)
        return height + config.row, width + config.column
