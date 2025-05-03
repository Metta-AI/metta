from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np

from mettagrid.map.node import Node
from mettagrid.map.scene import Scene
from mettagrid.map.utils.random import IntDistribution, MaybeSeed, sample_int_distribution

Anchor = Union[
    Literal["top-left"],
    Literal["top-right"],
    Literal["bottom-left"],
    Literal["bottom-right"],
]

ALL_ANCHORS: List[Anchor] = ["top-left", "top-right", "bottom-left", "bottom-right"]


def anchor_to_position(anchor: Anchor, width: int, height: int) -> Tuple[int, int]:
    if anchor == "top-left":
        return (0, 0)
    elif anchor == "top-right":
        return (width - 1, 0)
    elif anchor == "bottom-left":
        return (0, height - 1)
    elif anchor == "bottom-right":
        return (width - 1, height - 1)


class MazeKruskal(Scene):
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

    def __init__(
        self,
        room_size: IntDistribution = 1,
        wall_size: IntDistribution = 1,
        seed: MaybeSeed = None,
        children: Optional[List[Any]] = None,
    ):
        super().__init__(children=children)
        self._rng = np.random.default_rng(seed)
        self._room_size = sample_int_distribution(room_size, self._rng)
        self._wall_size = sample_int_distribution(wall_size, self._rng)

    def _render(self, node: Node):
        grid = node.grid
        width = node.width
        height = node.height
        room_size = self._room_size
        wall_size = self._wall_size
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

        self._rng.shuffle(walls)

        for wall in walls:
            col, row, direction = wall
            cell1 = (col, row)
            cell2 = (col, row + 1) if direction == "h" else (col + 1, row)
            if find(cell1) != find(cell2):
                clear_wall(wall)
                union(cell1, cell2)

        for anchor in ALL_ANCHORS:
            x, y = anchor_to_position(anchor, width, height)
            node.make_area(x, y, 1, 1, tags=[anchor])

        return grid
