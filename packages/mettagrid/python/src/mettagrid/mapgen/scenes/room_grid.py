from typing import Optional

from mettagrid.mapgen.scene import Scene, SceneConfig


class RoomGridConfig(SceneConfig):
    rows: Optional[int] = None
    columns: Optional[int] = None
    layout: Optional[list[list[str]]] = None

    # Default value guarantees that agents don't see beyond the walls.
    # Usually shouldn't be changed.
    border_width: int = 5

    border_object: str = "wall"


class RoomGrid(Scene[RoomGridConfig]):
    """
    Tile the scene with a grid of equally sized isolated rooms.

    This scene is destructive: it will overwrite the entire grid.

    Example when rows=2, columns=3, border_width=1:
    ┌────────────┐
    │   #   #   #│
    │   #   #   #│
    │############│
    │   #   #   #│
    │   #   #   #│
    └────────────┘

    Outer walls are drawn for the sake of the example readability. (They'll usually be provided by the container scene.)

    The right wall is there because rooms are equally sized, and there's some extra space on the right.

    By default, each room will be tagged with "room" and "room_{row}_{col}". If layout is provided,
    the tags will be taken from the layout instead; and in this case rows and columns will
    be inferred from the layout.
    """

    def post_init(self):
        config = self.config

        if config.layout is None:
            assert config.rows is not None and config.columns is not None, (
                "Either layout or rows and columns must be provided"
            )
            self._rows = config.rows
            self._columns = config.columns
        else:
            for row in config.layout:
                assert len(row) == len(config.layout[0]), "All rows must have the same number of columns"
            self._rows = len(config.layout)
            self._columns = len(config.layout[0])

    def _tags(self, row: int, col: int) -> list[str]:
        if self.config.layout is not None:
            return [self.config.layout[row][col]]
        else:
            return ["room", f"room_{row}_{col}"]

    def render(self):
        config = self.config
        room_width = (self.width - config.border_width * (self._columns - 1)) // self._columns
        room_height = (self.height - config.border_width * (self._rows - 1)) // self._rows

        # fill entire grid with walls
        self.grid[:] = config.border_object

        for row in range(self._rows):
            for col in range(self._columns):
                x = col * (room_width + config.border_width)
                y = row * (room_height + config.border_width)
                self.grid[y : y + room_height, x : x + room_width] = "empty"
                self.make_area(x, y, room_width, room_height, tags=self._tags(row, col))
