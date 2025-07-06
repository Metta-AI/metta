from dataclasses import dataclass
from typing import Literal, Union

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


@dataclass
class MazeGrid:
    grid: MapGrid
    room_size: int
    wall_size: int

    def __post_init__(self):
        (self.height, self.width) = self.grid.shape
        self.cols = (self.width + self.wall_size) // (self.room_size + self.wall_size)
        self.rows = (self.height + self.wall_size) // (self.room_size + self.wall_size)

    def cell_top_left(self, i: int, j: int) -> tuple[int, int]:
        return (
            i * (self.room_size + self.wall_size),
            j * (self.room_size + self.wall_size),
        )

    def neighbors(self, i: int, j: int) -> list[tuple[int, int]]:
        return [
            (i + di, j + dj)
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]
            if 0 <= i + di < self.cols and 0 <= j + dj < self.rows
        ]

    def remove_wall_between(self, i1: int, j1: int, i2: int, j2: int):
        rs = self.room_size
        ws = self.wall_size
        x1, y1 = self.cell_top_left(i1, j1)
        x2, y2 = self.cell_top_left(i2, j2)
        dx, dy = i2 - i1, j2 - j1
        if dx == 1:  # right neighbor
            self.grid[y1 : y1 + rs, x1 + rs : x1 + rs + ws] = "empty"
        elif dx == -1:  # left neighbor
            self.grid[y2 : y2 + rs, x2 + rs : x2 + rs + ws] = "empty"
        elif dy == 1:  # below
            self.grid[y1 + rs : y1 + rs + ws, x1 : x1 + rs] = "empty"
        elif dy == -1:  # above
            self.grid[y2 + rs : y2 + rs + ws, x2 : x2 + rs] = "empty"

    def carve_cell(self, i: int, j: int):
        x, y = self.cell_top_left(i, j)
        rs = self.room_size
        self.grid[y : y + rs, x : x + rs] = "empty"


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
        grid = self.grid
        width = self.width
        height = self.height
        room_size = self.maze.room_size
        wall_size = self.maze.wall_size
        rw_size = room_size + wall_size

        grid[:] = self.EMPTY

        v_wall_positions = [rw_size * i + room_size for i in range(self.maze.cols - 1)]
        h_wall_positions = [rw_size * i + room_size for i in range(self.maze.rows - 1)]

        for x in v_wall_positions:
            grid[:, x : x + wall_size] = self.WALL
        for y in h_wall_positions:
            grid[y : y + wall_size, :] = self.WALL

        # append virtual walls at the bottom and right of the grid
        v_wall_positions.append(width)
        h_wall_positions.append(height)

        cells = [(col, row) for row in range(self.maze.rows) for col in range(self.maze.cols)]

        # DSU
        parent = {cell: cell for cell in cells}

        def find(cell):
            if parent[cell] != cell:
                parent[cell] = find(parent[cell])
            return parent[cell]

        def union(c1, c2):
            parent[find(c2)] = find(c1)

        # all horizontal and vertical walls
        walls = [
            (col, row, "h")  # h is "horizontal wall", below the cell
            for col in range(self.maze.cols)
            for row in range(self.maze.rows - 1)
        ] + [
            (col, row, "v")  # v is "vertical wall", to the right of the cell
            for col in range(self.maze.cols - 1)
            for row in range(self.maze.rows)
        ]

        def clear_wall(wall):
            col, row, direction = wall
            if direction == "h":
                x0 = col * rw_size
                x1 = v_wall_positions[col]
                y0 = h_wall_positions[row]
                y1 = y0 + wall_size

                grid[y0:y1, x0:x1] = self.EMPTY
            else:
                x0 = v_wall_positions[col]
                x1 = x0 + wall_size
                y0 = row * rw_size
                y1 = h_wall_positions[row]

                grid[y0:y1, x0:x1] = self.EMPTY

        self.rng.shuffle(walls)

        for wall in walls:
            col, row, direction = wall
            cell1 = (col, row)
            cell2 = (col, row + 1) if direction == "h" else (col + 1, row)
            if find(cell1) != find(cell2):
                clear_wall(wall)
                union(cell1, cell2)

    def _render_dfs(self):
        # Initialize grid with walls and visited flags for maze cells.
        self.grid[:] = "wall"
        visited = np.zeros((self.maze.rows, self.maze.cols), dtype=bool)

        def carve_passages_from(i: int, j: int):
            nonlocal visited

            visited[j, i] = True
            self.maze.carve_cell(i, j)
            nbs = self.maze.neighbors(i, j)
            self.rng.shuffle(nbs)
            for ni, nj in nbs:
                if not visited[nj, ni]:
                    self.maze.remove_wall_between(i, j, ni, nj)
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
