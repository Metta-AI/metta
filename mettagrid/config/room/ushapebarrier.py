from typing import Any
import numpy as np
from omegaconf import OmegaConf, DictConfig
from mettagrid.config.room.room import Room

class UshapeBarrier(Room):
    """
    The 'UshapeBarrier' environment consists of a grid with a single room.

    In the room:
      - Walls are drawn along all borders.
      - The bottom wall has a door (a gap in the center).
      - The agent is placed in the center of the room's interior.
      - Immediately below the room (aligned with the door), three objects are placed:
          * The heart altar (directly below the door),
          * The generator (to the right of the door), and
          * The converter (to the left of the door).

    The overall grid dimensions and the room definition are provided via the configuration.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the UshapeBarrier environment using the provided configuration.
        
        The configuration should define:
          - grid: overall width and height,
          - border: the border width and the object used (e.g., "wall"),
          - rooms: a list of room definitions (each with keys: top, left, width, and height).
        """
        self.config = config
        self._width = config.grid.width
        self._height = config.grid.height
        self._border_width = config.border.width
        self._border_object = config.border.object
        self._rooms_config = config.rooms  # Expecting a list with one room

        # Initialize the grid with the border object.
        self._grid = np.full((self._height, self._width), self._border_object, dtype='<U50')

    def _build(self) -> np.ndarray:
        # Step 1: Fill the interior (excluding the outer border) with "empty"
        interior_x_start = self._border_width
        interior_x_end = self._width - self._border_width
        interior_y_start = self._border_width
        interior_y_end = self._height - self._border_width

        for r in range(interior_y_start, interior_y_end):
            for c in range(interior_x_start, interior_x_end):
                self._grid[r, c] = "empty"

        # Step 2: Build the room as defined in the configuration.
        # (Assuming one room is defined in self._rooms_config)
        for room in self._rooms_config:
            room_top = room.top
            room_left = room.left
            room_width = room.width
            room_height = room.height
            room_bottom = room_top + room_height - 1
            room_right = room_left + room_width - 1

            # Draw the top wall.
            for c in range(room_left, room_right + 1):
                self._grid[room_top, c] = "wall"

            # Draw the bottom wall with a door (a gap in the center).
            door_col = room_left + room_width // 2
            for c in range(room_left, room_right + 1):
                if c == door_col:
                    self._grid[room_bottom, c] = "empty"  # door gap
                else:
                    self._grid[room_bottom, c] = "wall"

            # Draw the left and right walls.
            for r in range(room_top, room_bottom + 1):
                self._grid[r, room_left] = "wall"
                self._grid[r, room_right] = "wall"

            # Step 3: Place the agent in the center of the room's interior.
            interior_room_top = room_top + 1
            interior_room_left = room_left + 1
            interior_room_bottom = room_bottom - 1
            interior_room_right = room_right - 1

            if interior_room_bottom >= interior_room_top and interior_room_right >= interior_room_left:
                agent_row = interior_room_top + (interior_room_bottom - interior_room_top) // 2
                agent_col = interior_room_left + (interior_room_right - interior_room_left) // 2
                self._grid[agent_row, agent_col] = "agent.agent"

            # Step 4: Place the objects below the room (aligned with the door)
            objects_row = room_bottom + 1
            if objects_row < interior_y_end:
                # Place the heart altar directly below the door.
                self._grid[objects_row, door_col] = "altar"
                # Place the generator to the right of the door.
                if door_col + 1 < interior_x_end:
                    self._grid[objects_row, door_col + 1] = "generator"
                # Place the converter to the left of the door.
                if door_col - 1 >= interior_x_start:
                    self._grid[objects_row, door_col - 1] = "converter"

        return self._grid

if __name__ == "__main__":
    # Sample configuration for testing purposes.
    config_dict = {
        "grid": {"width": 30, "height": 25},
        "border": {"width": 1, "object": "wall"},
        "rooms": [
            {"top": 5, "left": 5, "width": 7, "height": 5}
        ]
    }
    config = OmegaConf.create(config_dict)
    env = UshapeBarrier(config)
    grid = env._build()
    
    # Print the grid row by row for visualization.
    for row in grid:
        print(" ".join(row))
