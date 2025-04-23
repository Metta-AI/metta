"""
CubicleWorld
=============

Creates a cubicle of full‑length walls: vertical lines from top‑to‑bottom and
horizontal lines from left‑to‑right.  Interior rectangles of empty space are the
“cubicles.”  Door‑sized openings are carved in each interior wall.

The outermost perimeter is always a solid wall, ensuring the arena is fully enclosed.

Episode‑time random sampling
----------------------------
* env_width,  env_height  ← uniform int samples from width_range / height_range
* gap_x, gap_y            ← uniform int samples from gap_range (4‑14)

Wall coordinates:
    vertical walls at 0, gap_x+1, 2*(gap_x+1), …, width‑1
    horizontal walls at 0, gap_y+1, 2*(gap_y+1), …, height‑1
"""

from __future__ import annotations
from typing import Sequence
import numpy as np
from mettagrid.config.room.room import Room



class CubicleWorld(Room):
    STYLE_PARAMETERS = {"cubicle": {"hearts_count": 0}}

    def __init__(
        self,
        width_range: Sequence[int] = (120, 320),
        height_range: Sequence[int] = (120, 320),
        gap_range: Sequence[int] = (4, 14),
        altars_count: int = 50,
        agents: int | dict = 20,
        seed: int | None = 42,
        border_object: str = "wall",
        border_width: int = 6
    ):
        """
        Initialize a CubicleWorld environment.
        """
        super().__init__(border_width = border_width, border_object=border_object)
        rng = np.random.default_rng(seed)

        self.width = np.random.randint(width_range[0], width_range[1])
        self.height = np.random.randint(height_range[0], height_range[1])
        self.gap_x = np.random.randint(gap_range[0], gap_range[1])
        self.gap_y =  np.random.randint(gap_range[0], gap_range[1])

        self.altars_count = altars_count
        self._agents = agents
        self._rng = rng
        self.labels = ["cubicle"]
        self.set_size_labels(self.width, self.height)

    # --------------------------------------------------------------- #
    def _build(self):
        grid = self._generate_grid()
        self._place_altars(grid)
        self._place_agents(grid)
        return grid

    # --------------------------------------------------------------- #
    def _generate_grid(self) -> np.ndarray:
        H, W = self.height, self.width
        grid = np.full((H, W), "empty", dtype=object)

        # Draw vertical full‑height walls
        col = 0
        while col < W:
            grid[:, col] = "wall"
            col += self.gap_x + 1

        # Draw horizontal full‑width walls
        row = 0
        while row < H:
            grid[row, :] = "wall"
            row += self.gap_y + 1

        # Carve door‑sized breaks in every interior wall
        # Vertical walls: skip borders (col == 0 or col == W‑1)
        for col in range(self.gap_x + 1, W - 1, self.gap_x + 1):
            for row_start in range(0, H, self.gap_y + 1):
                if row_start + self.gap_y < H:
                    door_row = row_start + 1 + self._rng.integers(self.gap_y)
                    grid[door_row, col] = "empty"

        # Horizontal walls: skip borders (row == 0 or row == H‑1)
        for row in range(self.gap_y + 1, H - 1, self.gap_y + 1):
            for col_start in range(0, W, self.gap_x + 1):
                if col_start + self.gap_x < W:
                    door_col = col_start + 1 + self._rng.integers(self.gap_x)
                    grid[row, door_col] = "empty"

        # Ensure the outer perimeter is a solid wall
        grid[0, :] = "wall"
        grid[-1, :] = "wall"
        grid[:, 0] = "wall"
        grid[:, -1] = "wall"

        # Occupancy mask for placement
        self._occ = np.zeros((H, W), dtype=bool)
        return grid

    # --------------------------------------------------------------- #
    def _place_altars(self, grid: np.ndarray):
        empties = list(zip(*np.where(grid == "empty")))
        self._rng.shuffle(empties)
        for pos in empties[: self.altars_count]:
            grid[pos] = "altar"
            self._occ[pos] = True

    def _place_agents(self, grid: np.ndarray):
        # build tag list
        if isinstance(self._agents, int):
            tags = ["agent.agent"] * self._agents
        else:
            tags = ["agent." + t for t, n in self._agents.items() for _ in range(n)]

        empties = np.flatnonzero((grid == "empty") & (~self._occ))
        self._rng.shuffle(empties)
        for tag, idx in zip(tags, empties):
            pos = tuple(np.unravel_index(idx, grid.shape))
            grid[pos] = tag
            self._occ[pos] = True
