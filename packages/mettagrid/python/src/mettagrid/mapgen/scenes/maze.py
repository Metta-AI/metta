from dataclasses import dataclass
from typing import Literal, TypeAlias, Union

import numpy as np

from mettagrid.map_builder import MapGrid
from mettagrid.mapgen.random.int import IntConstantDistribution, IntDistribution
from mettagrid.mapgen.scene import Scene, SceneConfig

Anchor = Union[
    Literal["top-left"],
    Literal["top-right"],
    Literal["bottom-left"],
    Literal["bottom-right"],
]

ALL_ANCHORS: list[Anchor] = ["top-left", "top-right", "bottom-left", "bottom-right"]


def anchor_to_position(anchor: Anchor, width: int, height: int) -> tuple[int, int]:
    if anchor == "top-left":
        return (0, 0)
    elif anchor == "top-right":
        return (width - 1, 0)
    elif anchor == "bottom-left":
        return (0, height - 1)
    elif anchor == "bottom-right":
        return (width - 1, height - 1)


Direction: TypeAlias = tuple[int, int]

ALL_DIRECTIONS: list[Direction] = [(0, -1), (0, 1), (1, 0), (-1, 0)]


@dataclass
class MazeGrid:
    """
    A grid of maze cells of NxN size, separated by walls.

    Naming conventions:
    - `i` and `j` are the indices of the maze cells.
    - `x` and `y` are the coordinates in the underlying MapGrid.
    """

    grid: MapGrid
    room_size: int
    wall_size: int

    def __post_init__(self):
        (self.height, self.width) = self.grid.shape
        self.cols = (self.width + self.wall_size) // (self.room_size + self.wall_size)
        self.rows = (self.height + self.wall_size) // (self.room_size + self.wall_size)

    def cell_top_left(self, i: int, j: int) -> tuple[int, int]:
        """
        Returns the top-left corner of the cell at (i, j).
        """
        return (
            i * (self.room_size + self.wall_size),
            j * (self.room_size + self.wall_size),
        )

    def valid_directions(self, i: int, j: int) -> list[Direction]:
        return [d for d in ALL_DIRECTIONS if 0 <= i + d[0] < self.cols and 0 <= j + d[1] < self.rows]

    def _set_cell_border_in_direction(self, i1: int, j1: int, d: Direction, value: str):
        rs = self.room_size
        ws = self.wall_size
        (i2, j2) = (i1 + d[0], j1 + d[1])
        x1, y1 = self.cell_top_left(i1, j1)
        x2, y2 = self.cell_top_left(i2, j2)
        if d == (1, 0):  # right neighbor
            self.grid[y1 : y1 + rs, x1 + rs : x1 + rs + ws] = value
        elif d == (-1, 0):  # left neighbor
            self.grid[y2 : y2 + rs, x2 + rs : x2 + rs + ws] = value
        elif d == (0, 1):  # below
            self.grid[y1 + rs : y1 + rs + ws, x1 : x1 + rs] = value
        elif d == (0, -1):  # above
            self.grid[y2 + rs : y2 + rs + ws, x2 : x2 + rs] = value

    def remove_wall_in_direction(self, i1: int, j1: int, d: Direction):
        self._set_cell_border_in_direction(i1, j1, d, "empty")

    def carve_cell(self, i: int, j: int):
        x, y = self.cell_top_left(i, j)
        rs = self.room_size
        self.grid[y : y + rs, x : x + rs] = "empty"

    def clear_and_carve_all_cells(self):
        self.grid[:] = "empty"
        rw_size = self.room_size + self.wall_size

        for col in range(self.cols - 1):
            x = rw_size * col + self.room_size
            self.grid[:, x : x + self.wall_size] = "wall"
        for row in range(self.rows - 1):
            y = rw_size * row + self.room_size
            self.grid[y : y + self.wall_size, :] = "wall"


class MazeConfig(SceneConfig):
    algorithm: Literal["kruskal", "dfs"] = "kruskal"
    room_size: IntDistribution = IntConstantDistribution(value=1)
    wall_size: IntDistribution = IntConstantDistribution(value=1)


class Maze(Scene[MazeConfig]):
    """
    Maze generation scene.

    Supports two algorithms:
    1. `kruskal`: Kruskal's algorithm
    2. `dfs`: Depth-first search, recursive backtracking

    DFS algorithm tends to create longer, more winding corridors with fewer branches.

    The generated maze doesn't have an outer border.

    Example output:
    ┌─────────┐
    │         │
    │ # ##### │
    │ # # #   │
    │#### ### │
    │   # # # │
    │ ### # # │
    │         │
    │## ### ##│
    │     #   │
    └─────────┘
    """

    EMPTY, WALL = "empty", "wall"

    def post_init(self):
        # Calculate number of maze cells and adjust overall dimensions.
        sampled_room_size = self.config.room_size.sample(self.rng)
        room_size = max(1, min(sampled_room_size, self.width, self.height))

        sampled_wall_size = self.config.wall_size.sample(self.rng)
        wall_size = max(1, sampled_wall_size)

        self.maze = MazeGrid(self.grid, room_size, wall_size)

    def _render_kruskal(self):
        self.maze.clear_and_carve_all_cells()

        cols, rows = self.maze.cols, self.maze.rows

        parent = np.arange(cols * rows)

        def find(idx: int) -> int:
            root = idx
            while parent[root] != root:
                root = parent[root]
            while parent[idx] != root:
                idx, parent[idx] = parent[idx], root
            return root

        def union(i1: int, i2: int):
            r1, r2 = find(i1), find(i2)
            if r1 != r2:
                parent[r2] = r1

        walls = []
        for col in range(cols):
            for row in range(rows - 1):
                walls.append((col, row, (0, 1)))
        for col in range(cols - 1):
            for row in range(rows):
                walls.append((col, row, (1, 0)))

        self.rng.shuffle(walls)

        for col, row, direction in walls:
            idx1 = row * cols + col
            idx2 = (row + direction[1]) * cols + (col + direction[0])
            if find(idx1) != find(idx2):
                self.maze.remove_wall_in_direction(col, row, direction)
                union(idx1, idx2)

    def _render_dfs(self):
        # Initialize grid with walls and visited flags for maze cells.
        self.grid[:] = "wall"
        visited = np.zeros((self.maze.rows, self.maze.cols), dtype=bool)

        # Iterative DFS using explicit stack to avoid Python recursion limits.
        stack: list[tuple[int, int]] = [(0, 0)]
        visited[0, 0] = True
        self.maze.carve_cell(0, 0)

        while stack:
            i, j = stack[-1]
            # Find unvisited neighbors
            dirs = [d for d in self.maze.valid_directions(i, j) if not visited[j + d[1], i + d[0]]]
            if not dirs:
                stack.pop()
                continue

            # Randomly pick next direction
            d = dirs[int(self.rng.integers(0, len(dirs)))]
            ni, nj = i + d[0], j + d[1]
            self.maze.remove_wall_in_direction(i, j, d)
            visited[nj, ni] = True
            self.maze.carve_cell(ni, nj)
            stack.append((ni, nj))

    def render(self):
        if self.config.algorithm == "kruskal":
            self._render_kruskal()
        elif self.config.algorithm == "dfs":
            self._render_dfs()
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

        # Create clamped anchor sub-areas; avoid overflow in small zones
        for anchor in ALL_ANCHORS:
            i, j = anchor_to_position(anchor, self.maze.cols, self.maze.rows)
            x, y = self.maze.cell_top_left(i, j)
            w = max(1, min(self.maze.room_size, self.width - x))
            h = max(1, min(self.maze.room_size, self.height - y))
            if w > 0 and h > 0:
                self.make_area(x, y, w, h, tags=[anchor])
