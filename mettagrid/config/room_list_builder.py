import numpy as np
from omegaconf import ListConfig
from mettagrid.config.room_builder import RoomBuilder, SYMBOLS

class RoomListBuilder(RoomBuilder):
    def __init__(self, rooms: ListConfig, border_width: int = 0, border_object: str = "wall"):
        super().__init__(border_width=border_width, border_object=border_object)
        self._room_configs = rooms

    def _build(self):
        rooms = []

        max_height, max_width = 0, 0

        for room in self._room_configs:
            room_array = room.build()
            max_height = max(max_height, room_array.shape[0])
            max_width = max(max_width, room_array.shape[1])
            rooms.append(room_array)

        # Determine grid dimensions based on number of rooms
        n_rooms = len(rooms)

        grid_rows = int(np.ceil(np.sqrt(n_rooms)))
        grid_cols = int(np.ceil(n_rooms / grid_rows))

        # Create empty grid to hold all rooms
        level = np.full((grid_rows * max_height, grid_cols * max_width),
                      SYMBOLS["wall"], dtype="U6")

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
            level[start_row:start_row + room_height,
                 start_col:start_col + room_width] = rooms[i]

        return level
