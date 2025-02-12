from typing import Set, Tuple
import numpy as np
from omegaconf import DictConfig

from mettagrid.config.room.room import Room

class NestedRooms(Room):
    def __init__(
        self,
        width: int,
        height: int,
        num_nested: int,
        agents: int | DictConfig = 1,
        seed=None,
        border_width: int = 1,
        border_object: str = "wall",
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._width = width
        self._height = height
        self._num_nested = num_nested
        self._agents = agents if isinstance(agents, int) else agents
        self._border_width = border_width
        self._rng = np.random.default_rng(seed)
        self._grid = np.full((self._height, self._width), "empty", dtype='<U50')
        self._walls: Set[Tuple[int, int]] = set()
        self._agent_positions: Set[Tuple[int, int]] = set()

    def _build(self) -> np.ndarray:
        regions = []  # will hold the interior (non-wall) cells for each nested room
        # Define a gap between nested room borders.
        gap = self._border_width + 1

        for level in range(self._num_nested):
            offset = level * gap
            top = offset
            left = offset
            bottom = self._height - 1 - offset
            right = self._width - 1 - offset

            # If the rectangle would be invalid, stop drawing further nested rooms.
            if top > bottom or left > right:
                break

            # For every room except the outermost, create a door in the wall.
            door = None
            if level > 0:
                side = self._rng.choice(["top", "bottom", "left", "right"])
                if side == "top" and (right - left) > 2:
                    door_x = int(self._rng.integers(left + 1, right))
                    door = (door_x, top)
                elif side == "bottom" and (right - left) > 2:
                    door_x = int(self._rng.integers(left + 1, right))
                    door = (door_x, bottom)
                elif side == "left" and (bottom - top) > 2:
                    door_y = int(self._rng.integers(top + 1, bottom))
                    door = (left, door_y)
                elif side == "right" and (bottom - top) > 2:
                    door_y = int(self._rng.integers(top + 1, bottom))
                    door = (right, door_y)

            # Draw the border (i.e. the walls) of the nested room.
            for y in range(top, bottom + 1):
                for x in range(left, right + 1):
                    # If the cell is within the wall thickness along any edge…
                    if (
                        y < top + self._border_width
                        or y > bottom - self._border_width
                        or x < left + self._border_width
                        or x > right - self._border_width
                    ):
                        if door is not None and (x, y) == door:
                            self._grid[y, x] = "door"
                        else:
                            self._grid[y, x] = "wall"
                            self._walls.add((x, y))

            # Compute the interior region of this room (cells that remain "empty").
            interior = set()
            for y in range(top + self._border_width, bottom - self._border_width + 1):
                for x in range(left + self._border_width, right - self._border_width + 1):
                    if self._grid[y, x] == "empty":
                        interior.add((x, y))
            regions.append(interior)

        # --- Place special elements ---
        # We want to place a generator, a converter, and an altar in distinct nested regions if possible.
        items = ["generator", "converter", "altar"]
        if len(regions) >= len(items):
            chosen_levels = self._rng.choice(len(regions), size=len(items), replace=False)
        else:
            chosen_levels = list(range(len(regions)))
        for i, item in enumerate(items):
            # Only place if the chosen region has valid (empty) positions.
            if i < len(chosen_levels) and regions[chosen_levels[i]]:
                pos = self._rng.choice(list(regions[chosen_levels[i]]))
                self._grid[pos[1], pos[0]] = item

        # --- Place agents ---
        # Here we attempt to place agents in the innermost room.
        if regions:
            agent_region = regions[-1]
        else:
            # Fallback: use all non-wall cells.
            agent_region = {
                (x, y)
                for y in range(self._height)
                for x in range(self._width)
                if self._grid[y, x] == "empty"
            }
        agent_positions = list(agent_region)
        # If there aren’t enough positions in the innermost room, use all available regions.
        if len(agent_positions) < self._agents:
            union_region = set()
            for reg in regions:
                union_region |= reg
            agent_positions = list(union_region)
        if len(agent_positions) >= self._agents:
            chosen_agent_positions = self._rng.choice(agent_positions, size=self._agents, replace=False)
            for pos in chosen_agent_positions:
                self._grid[pos[1], pos[0]] = "agent.agent"
                self._agent_positions.add((pos[0], pos[1]))
        else:
            # If even that fails, place as many as possible.
            for pos in agent_positions[: self._agents]:
                self._grid[pos[1], pos[0]] = "agent.agent"
                self._agent_positions.add((pos[0], pos[1]))

        return self._grid
