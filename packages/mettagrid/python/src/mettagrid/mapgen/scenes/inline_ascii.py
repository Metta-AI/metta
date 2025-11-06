import numpy as np
import pydantic

import mettagrid.mapgen.scene
import mettagrid.mapgen.utils.ascii_grid


class InlineAsciiConfig(mettagrid.mapgen.scene.SceneConfig):
    data: str
    row: int = 0
    column: int = 0
    char_to_name: dict[str, str] = pydantic.Field(
        default_factory=mettagrid.mapgen.utils.ascii_grid.default_char_to_name
    )


class InlineAscii(mettagrid.mapgen.scene.Scene[InlineAsciiConfig]):
    def post_init(self):
        config = self.config

        lines, _, _ = mettagrid.mapgen.utils.ascii_grid.char_grid_to_lines(config.data)
        self.ascii_grid = np.array([list(line) for line in lines], dtype="U6")
        # Convert characters to object names using the provided mapping
        if config.char_to_name:
            self.ascii_grid = np.vectorize(lambda char: config.char_to_name.get(char, char))(self.ascii_grid)

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
        _, width, height = mettagrid.mapgen.utils.ascii_grid.char_grid_to_lines(config.data)
        return height + config.row, width + config.column
