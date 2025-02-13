from typing import Optional, Tuple, List
import numpy as np
from omegaconf import DictConfig
from mettagrid.config.room.room import Room

class LabyrinthMaze(Room):
    """
    Generates a labyrinth using recursive backtracking. Maze passages are blocks of size
    corridor_width×corridor_width separated by 1-cell walls. Overall grid dimensions are computed as:
        width  = (num_cell_cols * (corridor_width + 1)) + 1
        height = (num_cell_rows * (corridor_width + 1)) + 1

    After maze generation, three reward objects (generator, altar, converter) are placed in a small
    cluster near the center, and the agent is placed at the entrance (maze cell (0,0)).
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
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._desired_width = width
        self._desired_height = height
        self._corridor_width = corridor_width
        self._agents = agents
        self._rng = np.random.default_rng(seed)
        self._border_width = border_width
        self._border_object = border_object

        # Determine number of maze cells that fit into the desired dimensions.
        self._maze_cols = (width - 1) // (corridor_width + 1)
        self._maze_rows = (height - 1) // (corridor_width + 1)
        # Recompute overall dimensions to match the cell structure.
        self._width = self._maze_cols * (corridor_width + 1) + 1
        self._height = self._maze_rows * (corridor_width + 1) + 1

    def _cell_top_left(self, i: int, j: int) -> Tuple[int, int]:
        """Return overall grid coordinates of the top-left corner of maze cell (i, j)."""
        x = self._border_width + i * (self._corridor_width + 1)
        y = self._border_width + j * (self._corridor_width + 1)
        return (x, y)

    def _carve_cell(self, i: int, j: int):
        """Clear the corridor block for maze cell (i, j)."""
        x, y = self._cell_top_left(i, j)
        cw = self._corridor_width
        self._grid[y:y+cw, x:x+cw] = "empty"

    def _remove_wall_between(self, i1: int, j1: int, i2: int, j2: int):
        """Remove the wall between adjacent maze cells (i1,j1) and (i2,j2)."""
        cw = self._corridor_width
        x1, y1 = self._cell_top_left(i1, j1)
        x2, y2 = self._cell_top_left(i2, j2)
        dx, dy = i2 - i1, j2 - j1

        if dx == 1:  # right neighbor
            self._grid[y1:y1+cw, x1+cw] = "empty"
        elif dx == -1:  # left neighbor
            self._grid[y2:y2+cw, x2+cw] = "empty"
        elif dy == 1:  # below
            self._grid[y1+cw, x1:x1+cw] = "empty"
        elif dy == -1:  # above
            self._grid[y2+cw, x2:x2+cw] = "empty"

    def _neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Return valid neighbor cell indices for cell (i, j)."""
        nbs = []
        for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self._maze_cols and 0 <= nj < self._maze_rows:
                nbs.append((ni, nj))
        return nbs

    def _carve_passages_from(self, i: int, j: int):
        """Recursively carve passages from maze cell (i, j) using backtracking."""
        self._visited[j, i] = True
        self._carve_cell(i, j)
        nbs = self._neighbors(i, j)
        self._rng.shuffle(nbs)
        for ni, nj in nbs:
            if not self._visited[nj, ni]:
                self._remove_wall_between(i, j, ni, nj)
                self._carve_passages_from(ni, nj)

    def _build(self) -> np.ndarray:
        # Initialize grid with walls and a visited flag array for maze cells.
        self._grid = np.full((self._height, self._width), "wall", dtype='<U50')
        self._visited = np.zeros((self._maze_rows, self._maze_cols), dtype=bool)

        # Generate the maze starting at cell (0,0).
        self._carve_passages_from(0, 0)

        # Clear a rectangular region around the maze center for reward placement.
        center_x, center_y = self._width // 2, self._height // 2
        margin = 2
        start_row, end_row = max(0, center_y - margin), min(self._height, center_y + margin + 1)
        start_col = max(0, center_x - margin - 1)
        end_col = min(self._width, center_x + margin + 2)
        self._grid[start_row:end_row, start_col:end_col] = "empty"

        # Place reward objects in a horizontal line at the center.
        self._grid[center_y, center_x - 1] = "generator"
        self._grid[center_y, center_x]     = "altar"
        self._grid[center_y, center_x + 1] = "converter"

        # Place the agent at the entrance (cell (0,0)); position it at the center of its block.
        agent_x, agent_y = self._cell_top_left(0, 0)
        offset = self._corridor_width // 2
        self._grid[agent_y + offset, agent_x + offset] = "agent.agent"

        return self._grid
