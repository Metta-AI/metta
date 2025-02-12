from typing import Set, Tuple
import numpy as np
from omegaconf import DictConfig

# Assuming that Room is defined in mettagrid.config.room.room.
from mettagrid.config.room.room import Room

class RoomWithinRoom(Room):
    def __init__(
        self,
        width: int,
        height: int,
        inner_size_min: int,
        inner_size_max: int,
        inner_room_gap_min: int,
        inner_room_gap_max: int,
        border_width: int = 1,
        border_object: str = "wall",
        agents: int | DictConfig = 1,
        seed=None,
    ):
        """
        Creates an environment with an outer room and an inner room.

        The outer room is the overall grid, with walls on its border.
        The inner room's width and height are sampled uniformly between
        inner_size_min and inner_size_max, and it is centered within the outer room.
        The inner room has its own walls, except for a randomly sized door gap 
        (between inner_room_gap_min and inner_room_gap_max) in its top wall.

        Inside the inner room:
          - Top left: converter
          - Top right: altar
          - Bottom right: generator
          - Bottom left: agent

        In addition, a group of outer room objects (converter, altar, generator)
        is placed together in the top-left area of the overall grid.
        """
        super().__init__(border_width=border_width, border_object=border_object)
        self._overall_width = width
        self._overall_height = height
        self._inner_size_min = inner_size_min
        self._inner_size_max = inner_size_max
        self._inner_room_gap_min = inner_room_gap_min
        self._inner_room_gap_max = inner_room_gap_max
        self._agents = agents if isinstance(agents, int) else agents
        self._rng = np.random.default_rng(seed)
        self._grid = np.full((self._overall_height, self._overall_width), "empty", dtype='<U50')
        self._wall_positions: Set[Tuple[int, int]] = set()
        self._border_width = border_width

        # Sample inner room dimensions uniformly between inner_size_min and inner_size_max.
        self._inner_width = self._rng.integers(self._inner_size_min, self._inner_size_max + 1)
        self._inner_height = self._rng.integers(self._inner_size_min, self._inner_size_max + 1)

    def _build(self) -> np.ndarray:
        # --- Draw the outer room walls (around the entire grid) ---
        for y in range(self._overall_height):
            for x in range(self._overall_width):
                if (
                    x < self._border_width or x >= self._overall_width - self._border_width or 
                    y < self._border_width or y >= self._overall_height - self._border_width
                ):
                    self._grid[y, x] = self._border_object
                    self._wall_positions.add((x, y))
                    
        # --- Define the inner room dimensions (centered inside the outer room) ---
        inner_width = self._inner_width
        inner_height = self._inner_height

        inner_left = (self._overall_width - inner_width) // 2
        inner_top = (self._overall_height - inner_height) // 2
        inner_right = inner_left + inner_width - 1
        inner_bottom = inner_top + inner_height - 1

        # --- Determine the door (gap) size for the inner room's top wall ---
        door_gap = self._rng.integers(self._inner_room_gap_min, self._inner_room_gap_max + 1)
        # Ensure door_gap does not exceed the available length on the top wall (leaving the corners intact)
        max_gap_possible = inner_width - 2 * self._border_width - 2
        if max_gap_possible < door_gap:
            door_gap = max_gap_possible if max_gap_possible > 0 else door_gap
        # Center the door gap along the top wall (inside the wall border)
        door_start = inner_left + self._border_width + ((inner_width - 2 * self._border_width - door_gap) // 2)

        # --- Draw the inner room walls ---
        for y in range(inner_top, inner_bottom + 1):
            for x in range(inner_left, inner_right + 1):
                # Check if the cell is on the boundary of the inner room.
                if x == inner_left or x == inner_right or y == inner_top or y == inner_bottom:
                    # On the top wall, skip the door gap.
                    if y == inner_top and door_start <= x < door_start + door_gap:
                        self._grid[y, x] = "door"
                    else:
                        self._grid[y, x] = self._border_object
                        self._wall_positions.add((x, y))

        # --- Place objects inside the inner room (in its four corners) ---
        # Top left: converter
        inner_converter_pos = (inner_left + self._border_width, inner_top + self._border_width)
        self._grid[inner_converter_pos[1], inner_converter_pos[0]] = "converter"

        # Top right: altar
        inner_altar_pos = (inner_right - self._border_width, inner_top + self._border_width)
        self._grid[inner_altar_pos[1], inner_altar_pos[0]] = "altar"

        # Bottom right: generator
        inner_generator_pos = (inner_right - self._border_width, inner_bottom - self._border_width)
        self._grid[inner_generator_pos[1], inner_generator_pos[0]] = "generator"

        # Bottom left: agent
        inner_agent_pos = (inner_left + self._border_width, inner_bottom - self._border_width)
        self._grid[inner_agent_pos[1], inner_agent_pos[0]] = "agent.agent"

        # --- Place outer room objects as a grouped cluster in the top-left area ---
        # We choose cells just inside the outer wall (and outside the inner room, which is centered).
        outer_obj_x = self._border_width + 1
        outer_obj_y = self._border_width + 1

        # Place three items in adjacent cells: converter, altar, generator.
        self._grid[outer_obj_y, outer_obj_x] = "converter"
        if outer_obj_y + 1 < self._overall_height - self._border_width:
            self._grid[outer_obj_y + 1, outer_obj_x] = "altar"
        if outer_obj_x + 1 < self._overall_width - self._border_width:
            self._grid[outer_obj_y, outer_obj_x + 1] = "generator"

        return self._grid
