from typing import Set, Tuple
import numpy as np
from omegaconf import DictConfig
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

        The outer room is the overall grid with walls on its border.
        The inner room's dimensions are randomly sampled between inner_size_min
        and inner_size_max and centered within the outer room.
        Its top wall has a door gap (randomly sized between inner_room_gap_min
        and inner_room_gap_max). Specific objects are placed at the inner room's
        corners and a cluster of objects in the outer room's top-left.
        """
        super().__init__(border_width=border_width, border_object=border_object)
        self._overall_width, self._overall_height = width, height
        self._inner_size_min, self._inner_size_max = inner_size_min, inner_size_max
        self._inner_room_gap_min, self._inner_room_gap_max = inner_room_gap_min, inner_room_gap_max
        self._agents = agents if isinstance(agents, int) else agents
        self._rng = np.random.default_rng(seed)
        self._grid = np.full((height, width), "empty", dtype='<U50')
        self._wall_positions: Set[Tuple[int, int]] = set()
        self._border_width = border_width

        # Sample inner room dimensions uniformly between inner_size_min and inner_size_max.
        self._inner_width = self._rng.integers(inner_size_min, inner_size_max + 1)
        self._inner_height = self._rng.integers(inner_size_min, inner_size_max + 1)

    def _build(self) -> np.ndarray:
        bw, ow, oh = self._border_width, self._overall_width, self._overall_height

        # --- Draw outer room walls using slicing ---
        self._grid[:bw, :] = self._border_object
        self._grid[-bw:, :] = self._border_object
        self._grid[:, :bw] = self._border_object
        self._grid[:, -bw:] = self._border_object
        self._wall_positions.update(map(tuple, np.argwhere(self._grid == self._border_object)))

        # --- Define inner room dimensions (centered) ---
        inner_w, inner_h = self._inner_width, self._inner_height
        left = (ow - inner_w) // 2
        top = (oh - inner_h) // 2
        right = left + inner_w - 1
        bottom = top + inner_h - 1

        # --- Determine door gap on inner room's top wall ---
        door_gap = self._rng.integers(self._inner_room_gap_min, self._inner_room_gap_max + 1)
        max_gap = inner_w - 2 * bw - 2
        door_gap = min(door_gap, max_gap) if max_gap > 0 else door_gap
        door_start = left + bw + ((inner_w - 2 * bw - door_gap) // 2)

        # --- Draw inner room walls ---
        # Top wall (with door gap)
        for x in range(left, right + 1):
            if door_start <= x < door_start + door_gap:
                self._grid[top, x] = "door"
            else:
                self._grid[top, x] = self._border_object
                self._wall_positions.add((x, top))
        # Bottom wall
        for x in range(left, right + 1):
            self._grid[bottom, x] = self._border_object
            self._wall_positions.add((x, bottom))
        # Left and right walls (for interior rows)
        for y in range(top + 1, bottom):
            self._grid[y, left] = self._border_object
            self._wall_positions.add((left, y))
            self._grid[y, right] = self._border_object
            self._wall_positions.add((right, y))

        # --- Place inner room objects (corners) ---
        self._grid[top + bw, left + bw] = "converter"
        self._grid[top + bw, right - bw] = "altar"
        self._grid[bottom - bw, right - bw] = "generator"
        self._grid[bottom - bw, left + bw] = "agent.agent"

        # --- Place outer room objects (top-left cluster) ---
        ox, oy = bw + 1, bw + 1
        self._grid[oy, ox] = "converter"
        if oy + 1 < oh - bw:
            self._grid[oy + 1, ox] = "altar"
        if ox + 1 < ow - bw:
            self._grid[oy, ox + 1] = "generator"

        return self._grid
