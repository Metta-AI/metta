from typing import Any, Optional

from mettagrid.map.node import Node
from mettagrid.map.scene import Scene


class RoomGrid(Scene):
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

    Outer walls are drawn for the sake of the example readability. (They'll usually be provided by the container node.)

    The right wall is there because rooms are equally sized, and there's some extra space on the right.

    By default, each room will be tagged with "room" and "room_{row}_{col}". If layout is provided,
    the tags will be taken from the layout instead; and in this case rows and columns will
    be inferred from the layout.
    """

    def __init__(
        self,
        rows: Optional[int] = None,
        columns: Optional[int] = None,
        layout: Optional[list[list[str]]] = None,
        border_width: int = 1,
        border_object: str = "wall",
        children: Optional[list[Any]] = None,
    ):
        super().__init__(children=children)
        self._layout = layout
        if layout is None:
            assert rows is not None and columns is not None, "Either layout or rows and columns must be provided"
            self._rows = rows
            self._columns = columns
        else:
            for row in layout:
                assert len(row) == len(layout[0]), "All rows must have the same number of columns"
            self._rows = len(layout)
            self._columns = len(layout[0])

        self._border_width = border_width
        self._border_object = border_object

    def _tags(self, row: int, col: int) -> list[str]:
        if self._layout is not None:
            return [self._layout[row][col]]
        else:
            return ["room", f"room_{row}_{col}"]

    def _render(self, node: Node):
        room_width = (node.width - self._border_width * (self._columns - 1)) // self._columns
        room_height = (node.height - self._border_width * (self._rows - 1)) // self._rows

        # fill entire node.grid with walls
        node.grid[:] = self._border_object

        for row in range(self._rows):
            for col in range(self._columns):
                x = col * (room_width + self._border_width)
                y = row * (room_height + self._border_width)
                node.grid[y : y + room_height, x : x + room_width] = "empty"
                node.make_area(x, y, room_width, room_height, tags=self._tags(row, col))
