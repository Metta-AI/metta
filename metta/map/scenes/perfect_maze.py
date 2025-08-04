# metta/map/scenes/perfect_maze.py
from typing import Literal
import numpy as np
from metta.common.util.config import Config
from metta.map.scene import Scene
from metta.map.types import MapGrid

class PerfectMazeParams(Config):
    algorithm: Literal["kruskal", "recursive_backtracking", "prim"] = "recursive_backtracking"
    cell_size: int = 1  # Size of each cell in the maze
    wall_size: int = 1  # Size of walls between cells

class PerfectMaze(Scene[PerfectMazeParams]):
    """
    Generates a perfect maze where there's exactly one path between any two points.
    Optimized for performance with minimal overhead.
    """

    def render(self):
        # Calculate maze dimensions
        cell_size = self.params.cell_size
        wall_size = self.params.wall_size

        # Calculate number of cells that fit in the grid
        maze_width = (self.width - wall_size) // (cell_size + wall_size)
        maze_height = (self.height - wall_size) // (cell_size + wall_size)

        # Initialize with all walls
        self.grid[:] = "wall"

        if self.params.algorithm == "recursive_backtracking":
            self._recursive_backtracking(maze_width, maze_height)
        elif self.params.algorithm == "kruskal":
            self._kruskal(maze_width, maze_height)
        elif self.params.algorithm == "prim":
            self._prim(maze_width, maze_height)

    def _recursive_backtracking(self, maze_width: int, maze_height: int):
        """Fast recursive backtracking implementation"""
        cell_size = self.params.cell_size
        wall_size = self.params.wall_size

        # Track visited cells
        visited = np.zeros((maze_height, maze_width), dtype=bool)

        # Directions: up, right, down, left
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        def carve_path(cx: int, cy: int):
            visited[cy, cx] = True

            # Carve out the cell
            grid_x = cx * (cell_size + wall_size) + wall_size
            grid_y = cy * (cell_size + wall_size) + wall_size
            self.grid[grid_y:grid_y + cell_size, grid_x:grid_x + cell_size] = "empty"

            # Randomize directions
            dirs = self.rng.permutation(directions)

            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy

                if 0 <= nx < maze_width and 0 <= ny < maze_height and not visited[ny, nx]:
                    # Carve passage between cells
                    if dx == 1:  # Moving right
                        self.grid[grid_y:grid_y + cell_size,
                                 grid_x + cell_size:grid_x + cell_size + wall_size] = "empty"
                    elif dx == -1:  # Moving left
                        self.grid[grid_y:grid_y + cell_size,
                                 grid_x - wall_size:grid_x] = "empty"
                    elif dy == 1:  # Moving down
                        self.grid[grid_y + cell_size:grid_y + cell_size + wall_size,
                                 grid_x:grid_x + cell_size] = "empty"
                    elif dy == -1:  # Moving up
                        self.grid[grid_y - wall_size:grid_y,
                                 grid_x:grid_x + cell_size] = "empty"

                    carve_path(nx, ny)

        # Start from random cell
        start_x = self.rng.integers(0, maze_width)
        start_y = self.rng.integers(0, maze_height)
        carve_path(start_x, start_y)
