from typing import Optional, Tuple, List
import numpy as np
from omegaconf import DictConfig
from mettagrid.config.room.room import Room

class LabyrinthMaze(Room):
    """
    An environment that generates a labyrinth (maze) using recursive backtracking,
    with parameterizable corridor width and overall size.

    The maze is built on a grid that follows this pattern:
      - Walls are 1 cell thick.
      - Each maze cell (passage) is a block of size corridor_width × corridor_width.
      - The overall grid dimensions are computed as:
             width  = (num_cell_cols * (corridor_width + 1)) + 1
             height = (num_cell_rows * (corridor_width + 1)) + 1
    The recursive backtracking algorithm carves passages (by clearing the appropriate cells)
    between maze cells.
    
    After the maze is generated, three reward objects—**generator**, **altar**, and **converter**—
    are placed in a small cluster near the center of the maze.
    
    The agent is placed at the entrance (cell (0,0) in the maze cell space, corresponding to overall grid (1,1)).
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
        # Call the base constructor.
        super().__init__(border_width=border_width, border_object=border_object)
        self._desired_width = width
        self._desired_height = height
        self._corridor_width = corridor_width
        self._agents = agents
        self._rng = np.random.default_rng(seed)
        self._border_width = border_width
        self._border_object = border_object

        # Compute the number of maze cells (columns and rows) that fit into the desired dimensions.
        # The overall grid dimensions must satisfy: overall = (num_cells * (corridor_width+1)) + 1.
        self._maze_cols = (width - 1) // (corridor_width + 1)
        self._maze_rows = (height - 1) // (corridor_width + 1)
        # Recompute overall grid dimensions to exactly match the cell structure.
        self._width = self._maze_cols * (corridor_width + 1) + 1
        self._height = self._maze_rows * (corridor_width + 1) + 1

        # Initialize the grid with walls.
        self._grid = np.full((self._height, self._width), "wall", dtype='<U50')

        # We'll also create a visited array for maze cells.
        self._visited = np.zeros((self._maze_rows, self._maze_cols), dtype=bool)

    def _cell_top_left(self, i: int, j: int) -> Tuple[int, int]:
        """Return the overall grid coordinates of the top-left cell of maze cell (i, j)."""
        # The top-left of the first cell is at (border_width, border_width).
        # Each cell block has size corridor_width, and between cells there is 1 cell wall.
        x = self._border_width + i * (self._corridor_width + 1)
        y = self._border_width + j * (self._corridor_width + 1)
        return (x, y)

    def _carve_cell(self, i: int, j: int):
        """Clear out (set to empty) the corridor block for maze cell (i, j)."""
        x, y = self._cell_top_left(i, j)
        cw = self._corridor_width
        self._grid[y:y+cw, x:x+cw] = "empty"

    def _remove_wall_between(self, i1: int, j1: int, i2: int, j2: int):
        """Remove the wall between two adjacent maze cells (i1,j1) and (i2,j2)."""
        cw = self._corridor_width
        # Get top-left coordinates of both cells.
        x1, y1 = self._cell_top_left(i1, j1)
        x2, y2 = self._cell_top_left(i2, j2)
        # Determine the wall cell to remove.
        if i2 == i1 + 1:  # neighbor to the right
            wall_x = x1 + cw  # the column right after cell (i1, j1)'s block
            # Remove the wall in that column for the entire vertical extent of the corridor.
            self._grid[y1:y1+cw, wall_x] = "empty"
        elif i2 == i1 - 1:  # neighbor to the left
            wall_x = x2 + cw
            self._grid[y2:y2+cw, wall_x] = "empty"
        elif j2 == j1 + 1:  # neighbor below
            wall_y = y1 + cw
            self._grid[wall_y, x1:x1+cw] = "empty"
        elif j2 == j1 - 1:  # neighbor above
            wall_y = y2 + cw
            self._grid[wall_y, x2:x2+cw] = "empty"

    def _neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Return list of valid neighbor cell indices (i, j) for cell (i, j)."""
        neighbors = []
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self._maze_cols and 0 <= nj < self._maze_rows:
                neighbors.append((ni, nj))
        return neighbors

    def _carve_passages_from(self, i: int, j: int):
        self._visited[j, i] = True  # note: using (col, row) order: i is column index, j is row index.
        self._carve_cell(i, j)
        # Randomly order neighbors.
        nbs = self._neighbors(i, j)
        self._rng.shuffle(nbs)
        for ni, nj in nbs:
            if not self._visited[nj, ni]:
                self._remove_wall_between(i, j, ni, nj)
                self._carve_passages_from(ni, nj)

    def _build(self) -> np.ndarray:
        # --- Generate the Maze ---
        # Start at cell (0,0) as the entrance.
        self._carve_passages_from(0, 0)

        # --- Clear a region around the center for rewards ---
        # Compute the center of the overall grid.
        center_x = self._width // 2
        center_y = self._height // 2

        # Define a margin (in grid cells) to ensure empty space around the rewards.
        margin = 2  # Adjust this value for more/less clearance.

        # Compute the rectangular region to clear.
        start_row = max(0, center_y - margin)
        end_row = min(self._height, center_y + margin + 1)
        start_col = max(0, center_x - 1 - margin)
        end_col = min(self._width, center_x + 1 + margin + 1)

        # Clear the defined region (set to "empty").
        self._grid[start_row:end_row, start_col:end_col] = "empty"

        # --- Place the Reward Objects at the Maze Center ---
        # Place the generator, altar, and converter in a horizontal line.
        self._grid[center_y, center_x - 1] = "generator"
        self._grid[center_y, center_x] = "altar"
        self._grid[center_y, center_x + 1] = "converter"

        # --- Place the Agent at the Entrance ---
        # The entrance is cell (0,0) whose passage block starts at overall (1,1).
        agent_x, agent_y = self._cell_top_left(0, 0)
        # Place the agent marker in the middle of that passage block.
        self._grid[agent_y + self._corridor_width // 2, agent_x + self._corridor_width // 2] = "agent.agent"

        return self._grid
