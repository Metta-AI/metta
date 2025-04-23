"""
CubicleUniformWorld
===================
Variant of *CubicleWorld* in which every interior wall has a single door
**exactly at the midpoint** of the wall segment, producing perfectly aligned
corridors between adjacent cubicles.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np

from mettagrid.config.room.room import Room


class CubicleUniformWorld(Room):
    STYLE_PARAMETERS = {"cubicle_uniform": {"hearts_count": 0}}

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        width_range: Sequence[int] = (120, 320),
        height_range: Sequence[int] = (120, 320),
        gap_range: Sequence[int] = (4, 14),
        altars_count: int = 50,
        agents: int | dict = 20,
        seed: int | None = 42,
        border_object: str = "wall",
    ):
        super().__init__(border_object=border_object)
        rng = np.random.default_rng(seed)

        self.width = rng.integers(width_range[0], width_range[1])
        self.height = rng.integers(height_range[0], height_range[1])
        self.gap_x = rng.integers(gap_range[0], gap_range[1])
        self.gap_y = rng.integers(gap_range[0], gap_range[1])

        self.altars_count = altars_count
        self._agents = agents
        self._rng = rng
        self.labels = ["cubicle_uniform"]
        self.set_size_labels(self.width, self.height)
    # ------------------------------------------------------------------ #
    def _build(self):
        grid = self._generate_grid()
        self._place_altars(grid)
        self._place_agents(grid)
        return grid

    # ------------------------------------------------------------------ #
    def _generate_grid(self) -> np.ndarray:
        """Create the cubicle lattice with centred doors."""
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

        # Carve centred doors in vertical walls (skip outer perimeter)
        for col in range(self.gap_x + 1, W - 1, self.gap_x + 1):
            row_start = 0
            while row_start + self.gap_y < H:
                door_row = row_start + 1 + self.gap_y // 2
                grid[door_row, col] = "empty"
                row_start += self.gap_y + 1

        # Carve centred doors in horizontal walls
        for row in range(self.gap_y + 1, H - 1, self.gap_y + 1):
            col_start = 0
            while col_start + self.gap_x < W:
                door_col = col_start + 1 + self.gap_x // 2
                grid[row, door_col] = "empty"
                col_start += self.gap_x + 1

        # Ensure solid outer perimeter
        grid[0, :] = grid[-1, :] = "wall"
        grid[:, 0] = grid[:, -1] = "wall"

        # Occupancy mask
        self._occ = np.zeros((H, W), dtype=bool)
        return grid

    # ------------------------------------------------------------------ #
    def _place_altars(self, grid: np.ndarray):
        empties = list(zip(*np.where(grid == "empty")))
        self._rng.shuffle(empties)
        for pos in empties[: self.altars_count]:
            grid[pos] = "altar"
            self._occ[pos] = True

    def _place_agents(self, grid: np.ndarray):
        tags = (
            ["agent.agent"] * self._agents
            if isinstance(self._agents, int)
            else ["agent." + t for t, n in self._agents.items() for _ in range(n)]
        )
        empties = np.flatnonzero((grid == "empty") & (~self._occ))
        self._rng.shuffle(empties)
        for tag, idx in zip(tags, empties):
            pos = tuple(np.unravel_index(idx, grid.shape))
            grid[pos] = tag
            self._occ[pos] = True
