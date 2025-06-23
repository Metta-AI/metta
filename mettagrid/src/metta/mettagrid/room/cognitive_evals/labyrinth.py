from typing import List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room


class LabyrinthMaze(Room):
    """
    Generates a labyrinth using recursive backtracking. Maze passages are corridor_width×corridor_width blocks
    separated by 1-cell walls. The grid dimensions are computed to exactly fit the maze cells.
    Rewards (mine, altar, generator) are placed near the center and the agent at the entrance.
    """

    def __init__(
        self,
        width: int,
        height: int,
        corridor_width: int = 3,
        agents: int | DictConfig = 1,
        seed: Optional[int] = None,
        border_width: int = 1,
        border_object: str = "wall",
        onlyhearts: bool = False,
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._desired_width, self._desired_height = width, height
        self._corridor_width = corridor_width
        self._agents = agents
        self._rng = np.random.default_rng(seed)
        self._border_width = border_width
        self._border_object = border_object
        self._onlyhearts = onlyhearts

        # Calculate number of maze cells and adjust overall dimensions.
        self._maze_cols = (width - 1) // (corridor_width + 1)
        self._maze_rows = (height - 1) // (corridor_width + 1)
        self._width = self._maze_cols * (corridor_width + 1) + 1
        self._height = self._maze_rows * (corridor_width + 1) + 1

    def _cell_top_left(self, i: int, j: int) -> Tuple[int, int]:
        return (
            self._border_width + i * (self._corridor_width + 1),
            self._border_width + j * (self._corridor_width + 1),
        )

    def _carve_cell(self, i: int, j: int):
        x, y = self._cell_top_left(i, j)
        cw = self._corridor_width
        self._grid[y : y + cw, x : x + cw] = "empty"

    def _remove_wall_between(self, i1: int, j1: int, i2: int, j2: int):
        cw = self._corridor_width
        x1, y1 = self._cell_top_left(i1, j1)
        x2, y2 = self._cell_top_left(i2, j2)
        dx, dy = i2 - i1, j2 - j1
        if dx == 1:  # right neighbor
            self._grid[y1 : y1 + cw, x1 + cw] = "empty"
        elif dx == -1:  # left neighbor
            self._grid[y2 : y2 + cw, x2 + cw] = "empty"
        elif dy == 1:  # below
            self._grid[y1 + cw, x1 : x1 + cw] = "empty"
        elif dy == -1:  # above
            self._grid[y2 + cw, x2 : x2 + cw] = "empty"

    def _neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        return [
            (i + di, j + dj)
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]
            if 0 <= i + di < self._maze_cols and 0 <= j + dj < self._maze_rows
        ]

    def _carve_passages_from(self, i: int, j: int):
        self._visited[j, i] = True
        self._carve_cell(i, j)
        nbs = self._neighbors(i, j)
        self._rng.shuffle(nbs)
        for ni, nj in nbs:
            if not self._visited[nj, ni]:
                self._remove_wall_between(i, j, ni, nj)
                self._carve_passages_from(ni, nj)

    def _build(self) -> np.ndarray:
        # Initialize grid with walls and visited flags for maze cells.
        wall_type = "wall"
        self._grid = np.full((self._height, self._width), wall_type, dtype="<U50")
        self._visited = np.zeros((self._maze_rows, self._maze_cols), dtype=bool)
        self._carve_passages_from(0, 0)

        # Clear a small rectangle around the maze center for reward placement.
        center_x, center_y = self._width // 2, self._height // 2
        margin = 2
        self._grid[
            max(0, center_y - margin) : min(self._height, center_y + margin + 1),
            max(0, center_x - margin - 1) : min(self._width, center_x + margin + 2),
        ] = "empty"

        # Place objects, using hearts if onlyhearts is True
        object_type = "altar" if self._onlyhearts else "mine"
        self._grid[center_y, center_x - 1] = object_type
        object_type = "altar" if self._onlyhearts else "altar"
        self._grid[center_y, center_x] = object_type
        object_type = "altar" if self._onlyhearts else "generator"
        self._grid[center_y, center_x + 1] = object_type

        # Place the agent at the entrance (maze cell (0,0)), centered in its block.
        agent_x, agent_y = self._cell_top_left(0, 0)
        offset = self._corridor_width // 2
        self._grid[agent_y + offset, agent_x + offset] = "agent.agent"
        return self._grid
