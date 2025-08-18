from typing import Optional

from metta.common.util.config import Config
from metta.map.scene import Scene


class RoomGridParams(Config):
    rows: Optional[int] = None
    columns: Optional[int] = None
    layout: Optional[list[list[str]]] = None

    # Default value guarantees that agents don't see beyond the walls.
    # Usually shouldn't be changed.
    border_width: int = 5

    border_object: str = "wall"


class RoomGrid(Scene[RoomGridParams]):
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
        params = self.params

        if params.layout is None:
            assert params.rows is not None and params.columns is not None, (
                "Either layout or rows and columns must be provided"
            )
            self._rows = params.rows
            self._columns = params.columns
        else:
            for row in params.layout:
                assert len(row) == len(params.layout[0]), "All rows must have the same number of columns"
            self._rows = len(params.layout)
            self._columns = len(params.layout[0])

    def _tags(self, row: int, col: int) -> list[str]:
        if self.params.layout is not None:
            return [self.params.layout[row][col]]
        else:
            return ["room", f"room_{row}_{col}"]

    def render(self):
        params = self.params
        room_width = (self.width - params.border_width * (self._columns - 1)) // self._columns
        room_height = (self.height - params.border_width * (self._rows - 1)) // self._rows

        # fill entire grid with walls
        self.grid[:] = params.border_object

        for row in range(self._rows):
            for col in range(self._columns):
                x = col * (room_width + params.border_width)
                y = row * (room_height + params.border_width)
                self.grid[y : y + room_height, x : x + room_width] = "empty"
                self.make_area(x, y, room_width, room_height, tags=self._tags(row, col))

    def get_labels(self) -> list[str]:
        # Note: this code is from `metta.mettagrid.room.room_list`.
        # In case of mapgen, it's not very reliable, because any new child
        # scene, e.g. `make_connected`, would lead to zero common labels.
        room_labels: list[list[str]] = []

        for child_scene in self.children:
            # how do we want to account for room lists with different labels?
            room_labels.append(child_scene.get_labels())

        if not room_labels:
            return []

        common_labels = set.intersection(*[set(labels) for labels in room_labels])
        return list(common_labels)
