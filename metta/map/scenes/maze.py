from dataclasses import dataclass
from typing import Literal, TypeAlias, Union

import numpy as np

from metta.common.util.config import Config
from metta.map.random.int import IntConstantDistribution, IntDistribution
from metta.map.scene import Scene
from metta.map.types import MapGrid

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


class MazeParams(Config):
    algorithm: Literal["kruskal", "dfs"] = "kruskal"
    room_size: IntDistribution = IntConstantDistribution(value=1)
    wall_size: IntDistribution = IntConstantDistribution(value=1)


class Maze(Scene[MazeParams]):
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
        room_size = self.params.room_size.sample(self.rng)
        wall_size = self.params.wall_size.sample(self.rng)

        self.maze = MazeGrid(self.grid, room_size, wall_size)

    def _render_kruskal(self):
        self.maze.clear_and_carve_all_cells()

        cells = [(col, row) for row in range(self.maze.rows) for col in range(self.maze.cols)]

        # DSU
        parent = {cell: cell for cell in cells}

        def find(cell):
            if parent[cell] != cell:
                parent[cell] = find(parent[cell])
            return parent[cell]

        def union(c1, c2):
            parent[find(c2)] = find(c1)

        # all horizontal and vertical walls, expressed as a list of tuples (col, row, direction)
        walls: list[tuple[int, int, Direction]] = [
            (col, row, (0, 1))  # direction between cells is "down", this is a horizontal wall
            for col in range(self.maze.cols)
            for row in range(self.maze.rows - 1)
        ] + [
            (col, row, (1, 0))  # direction between cells is "right", this is a vertical wall
            for col in range(self.maze.cols - 1)
            for row in range(self.maze.rows)
        ]

        self.rng.shuffle(walls)

        for wall in walls:
            col, row, direction = wall
            cell1 = (col, row)
            cell2 = (col + direction[0], row + direction[1])
            if find(cell1) != find(cell2):
                self.maze.remove_wall_in_direction(col, row, direction)
                union(cell1, cell2)

    def _render_dfs(self):
        # Initialize grid with walls and visited flags for maze cells.
        self.grid[:] = "wall"
        visited = np.zeros((self.maze.rows, self.maze.cols), dtype=bool)

        def carve_passages_from(i: int, j: int):
            nonlocal visited

            visited[j, i] = True
            self.maze.carve_cell(i, j)
            directions = self.maze.valid_directions(i, j)
            self.rng.shuffle(directions)
            for d in directions:
                ni, nj = i + d[0], j + d[1]
                if not visited[nj, ni]:
                    self.maze.remove_wall_in_direction(i, j, d)
                    carve_passages_from(ni, nj)

        carve_passages_from(0, 0)

    def render(self):
        if self.params.algorithm == "kruskal":
            self._render_kruskal()
        elif self.params.algorithm == "dfs":
            self._render_dfs()
        else:
            raise ValueError(f"Unknown algorithm: {self.params.algorithm}")

        for anchor in ALL_ANCHORS:
            i, j = anchor_to_position(anchor, self.maze.cols, self.maze.rows)
            x, y = self.maze.cell_top_left(i, j)
            self.make_area(x, y, self.maze.room_size, self.maze.room_size, tags=[anchor])
