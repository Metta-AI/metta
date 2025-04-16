from typing import Any, List

from mettagrid.map.scene import Scene
from mettagrid.map.node import Node


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
    """

    def __init__(
        self,
        rows: int,
        columns: int,
        border_width: int = 1,
        border_object: str = "wall",
        children: List[Any] = [],
    ):
        super().__init__(children=children)
        self._rows = rows
        self._columns = columns
        self._border_width = border_width
        self._border_object = border_object

    def _render(self, node: Node):
        room_width = (
            node.width - self._border_width * (self._columns - 1)
        ) // self._columns
        room_height = (
            node.height - self._border_width * (self._rows - 1)
        ) // self._rows

        # fill entire node.grid with walls
        node.grid[:] = self._border_object

        for row in range(self._rows):
            for col in range(self._columns):
                x = col * (room_width + self._border_width)
                y = row * (room_height + self._border_width)
                node.grid[y : y + room_height, x : x + room_width] = "empty"
                node.make_area(x, y, room_width, room_height, tags=["room"])
