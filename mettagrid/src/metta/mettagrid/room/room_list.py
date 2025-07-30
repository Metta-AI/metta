from typing import List

import numpy as np

from metta.mettagrid.room.room import Room


class RoomList(Room):
    def __init__(self, rooms: List[Room], layout: str = "grid", border_width: int = 0, border_object: str = "wall"):
        super().__init__(border_width=border_width, border_object=border_object)
        self._room_configs = rooms
        self._layout = layout
        assert self._layout in ["grid", "column", "row"], (
            f"Invalid layout: {self._layout}. Must be 'grid', 'column', or 'row'"
        )

    def _build(self):
        rooms = []

        max_height, max_width = 0, 0

        room_labels = []

        for room in self._room_configs:
            room_level = room.build()
            grid = room_level.grid
            max_height = max(max_height, grid.shape[0])
            max_width = max(max_width, grid.shape[1])
            rooms.append(grid)
            # how do we want to account for room lists with different labels?
            room_labels.append(room_level.labels)

        # Find overlapping labels between all rooms
        common_labels = set.intersection(*[set(labels) for labels in room_labels])
        self.labels = list(common_labels)

        # Determine grid dimensions based on number of rooms
        n_rooms = len(rooms)

        grid_cols, grid_rows = 1, 1
        if self._layout == "grid":
            grid_rows = int(np.ceil(np.sqrt(n_rooms)))
            grid_cols = int(np.ceil(n_rooms / grid_rows))
        elif self._layout == "column":
            grid_rows = n_rooms
        elif self._layout == "row":
            grid_cols = n_rooms

        # Create empty grid to hold all rooms
        level = np.full((grid_rows * max_height, grid_cols * max_width), "empty", dtype="<U50")

        # Place rooms into grid
        for i in range(n_rooms):
            row = i // grid_cols
            col = i % grid_cols
            room_height = rooms[i].shape[0]
            room_width = rooms[i].shape[1]

            # Calculate starting position to center the room in its grid cell
            start_row = row * max_height + (max_height - room_height) // 2
            start_col = col * max_width + (max_width - room_width) // 2

            # Place room in centered position
            level[start_row : start_row + room_height, start_col : start_col + room_width] = rooms[i]

        return level
