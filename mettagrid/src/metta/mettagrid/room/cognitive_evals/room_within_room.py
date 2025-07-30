from typing import Set, Tuple

import numpy as np

from metta.mettagrid.room.room import Room
from metta.mettagrid.room.utils import create_grid, draw_border  # New utility functions


class RoomWithinRoom(Room):
    """
    Outer room with walls and a centered inner room (with a door gap in its top wall).
    Specific objects are placed at the inner room corners and in the outer room's top-left.
    """

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
        agents: int = 1,
        seed=None,
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._overall_width, self._overall_height = width, height
        self._inner_size_min, self._inner_size_max = inner_size_min, inner_size_max
        self._inner_room_gap_min, self._inner_room_gap_max = inner_room_gap_min, inner_room_gap_max
        self._agents = agents
        self._rng = np.random.default_rng(seed)
        self._wall_positions: Set[Tuple[int, int]] = set()
        self._border_width = border_width

        # Sample inner room dimensions.
        self._inner_width = self._rng.integers(inner_size_min, inner_size_max + 1)
        self._inner_height = self._rng.integers(inner_size_min, inner_size_max + 1)

    def _build(self) -> np.ndarray:
        grid = create_grid(self._overall_height, self._overall_width, fill_value="empty")
        bw, ow, oh = self._border_width, self._overall_width, self._overall_height

        # Draw outer walls using our utility.
        draw_border(grid, bw, self._border_object)
        self._wall_positions.update(map(tuple, np.argwhere(grid == self._border_object)))

        # Define inner room dimensions (centered).
        inner_w, inner_h = self._inner_width, self._inner_height
        left = (ow - inner_w) // 2
        top = (oh - inner_h) // 2
        right = left + inner_w - 1
        bottom = top + inner_h - 1

        # Determine door gap on the inner room's top wall.
        door_gap = self._rng.integers(self._inner_room_gap_min, self._inner_room_gap_max + 1)
        max_gap = inner_w - 2 * bw - 2
        door_gap = min(door_gap, max_gap) if max_gap > 0 else door_gap
        door_start = left + bw + ((inner_w - 2 * bw - door_gap) // 2)

        # Draw inner room walls.
        for x in range(left, right + 1):
            if door_start <= x < door_start + door_gap:
                grid[top, x] = "door"
            else:
                grid[top, x] = self._border_object
                self._wall_positions.add((x, top))
        for x in range(left, right + 1):
            grid[bottom, x] = self._border_object
            self._wall_positions.add((x, bottom))
        for y in range(top + 1, bottom):
            grid[y, left] = self._border_object
            grid[y, right] = self._border_object
            self._wall_positions.add((left, y))
            self._wall_positions.add((right, y))

        # Place inner room objects at the corners.
        grid[top + bw, left + bw] = "generator"
        grid[top + bw, right - bw] = "altar"
        grid[bottom - bw, right - bw] = "mine"
        grid[bottom - bw, left + bw] = "agent.agent"

        # Place outer room objects (top-left cluster).
        ox, oy = bw + 1, bw + 1
        grid[oy, ox] = "generator"
        if oy + 1 < oh - bw:
            grid[oy + 1, ox] = "altar"
        if ox + 1 < ow - bw:
            grid[oy, ox + 1] = "mine"

        return grid
