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


class MazeParams(Config):
    room_size: IntDistribution = IntConstantDistribution(value=1)
    wall_size: IntDistribution = IntConstantDistribution(value=1)


class MazeKruskal(Scene[MazeParams]):
    """
    Maze generation using Randomized Kruskal's algorithm.

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

    def render(self):
        grid = self.grid
        width = self.width
        height = self.height
        room_size = self.params.room_size.sample(self.rng)
        wall_size = self.params.wall_size.sample(self.rng)
        rw_size = room_size + wall_size

        width_in_rooms = (width + wall_size) // rw_size
        height_in_rooms = (height + wall_size) // rw_size

        grid[:] = self.EMPTY

        v_wall_positions = [rw_size * i + room_size for i in range(width_in_rooms - 1)]
        h_wall_positions = [rw_size * i + room_size for i in range(height_in_rooms - 1)]

        for x in v_wall_positions:
            grid[:, x : x + wall_size] = self.WALL
        for y in h_wall_positions:
            grid[y : y + wall_size, :] = self.WALL

        # append virtual walls at the bottom and right of the grid
        v_wall_positions.append(width)
        h_wall_positions.append(height)

        cells = [(col, row) for row in range(height_in_rooms) for col in range(width_in_rooms)]

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
            for col in range(width_in_rooms)
            for row in range(height_in_rooms - 1)
        ] + [
            (col, row, "v")  # v is "vertical wall", to the right of the cell
            for col in range(width_in_rooms - 1)
            for row in range(height_in_rooms)
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

        for anchor in ALL_ANCHORS:
            x, y = anchor_to_position(anchor, width, height)
            self.make_area(x, y, 1, 1, tags=[anchor])


@dataclass
class MazeGrid:
    grid: MapGrid
    corridor_width: int
    border_width: int

    def __post_init__(self):
        (self.height, self.width) = self.grid.shape
        self.cols = (self.width + self.border_width) // (self.corridor_width + self.border_width)
        self.rows = (self.height + self.border_width) // (self.corridor_width + self.border_width)

    def cell_top_left(self, i: int, j: int) -> tuple[int, int]:
        return (
            i * (self.corridor_width + self.border_width),
            j * (self.corridor_width + self.border_width),
        )

    def neighbors(self, i: int, j: int) -> list[tuple[int, int]]:
        return [
            (i + di, j + dj)
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]
            if 0 <= i + di < self.cols and 0 <= j + dj < self.rows
        ]

    def remove_wall_between(self, i1: int, j1: int, i2: int, j2: int):
        cw = self.corridor_width
        bw = self.border_width
        x1, y1 = self.cell_top_left(i1, j1)
        x2, y2 = self.cell_top_left(i2, j2)
        dx, dy = i2 - i1, j2 - j1
        if dx == 1:  # right neighbor
            self.grid[y1 : y1 + cw, x1 + cw : x1 + cw + bw] = "empty"
        elif dx == -1:  # left neighbor
            self.grid[y2 : y2 + cw, x2 + cw : x2 + cw + bw] = "empty"
        elif dy == 1:  # below
            self.grid[y1 + cw : y1 + cw + bw, x1 : x1 + cw] = "empty"
        elif dy == -1:  # above
            self.grid[y2 + cw : y2 + cw + bw, x2 : x2 + cw] = "empty"

    def carve_cell(self, i: int, j: int):
        x, y = self.cell_top_left(i, j)
        cw = self.corridor_width
        self.grid[y : y + cw, x : x + cw] = "empty"


class MazeLabyrinth(Scene[MazeParams]):
    """
    Maze generation using recursive backtracking.
    """

    def post_init(self):
        # Calculate number of maze cells and adjust overall dimensions.
        corridor_width = self.params.room_size.sample(self.rng)
        border_width = self.params.wall_size.sample(self.rng)

        self.maze = MazeGrid(self.grid, corridor_width, border_width)

    def render(self):
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

        for anchor in ALL_ANCHORS:
            i, j = anchor_to_position(anchor, self.maze.width, self.maze.height)
            x, y = self.maze.cell_top_left(i, j)
            self.make_area(x, y, self.maze.corridor_width, self.maze.corridor_width, tags=[anchor])
