"""
SquareWorld
===========

Creates a regular lattice of *solid‑wall squares* separated by uniform
corridors:

* **square_size ∈ [3 .. 7]** ─ edge length of every solid square block
* **gap ∈ [3 .. 6]**         ─ corridor width between squares (and between
  the squares and the outer perimeter)

Both parameters are sampled independently each episode.
A solid perimeter wall encloses the arena.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np

from mettagrid.config.room.room import Room


class SquareWorld(Room):
    STYLE_PARAMETERS = {"square_grid": {"hearts_count": 0}}

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        width_range: Sequence[int] = (60, 140),
        height_range: Sequence[int] = (60, 140),
        square_size_range: Sequence[int] = (3, 7),
        gap_range: Sequence[int] = (3, 6),
        altars_count: int = 50,
        agents: int | dict = 20,
        seed: int | None = 42,
        border_object: str = "wall",
    ):
        super().__init__(border_object=border_object)
        rng = np.random.default_rng(seed)

        self.square_size = rng.integers(square_size_range[0], square_size_range[1] + 1)
        self.gap = rng.integers(gap_range[0], gap_range[1] + 1)

        self.width = rng.integers(width_range[0], width_range[1] + 1)
        self.height = rng.integers(height_range[0], height_range[1] + 1)

        self.altars_count = altars_count
        self._agents = agents
        self._rng = rng
        self.labels = ["square_grid"]
        self.set_size_labels(self.width, self.height)

    # ------------------------------------------------------------------ #
    def _build(self):
        grid = self._generate_grid()
        self._place_altars(grid)
        self._place_agents(grid)
        return grid

    # ------------------------------------------------------------------ #
    def _generate_grid(self) -> np.ndarray:
        H, W = self.height, self.width
        grid = np.full((H, W), "empty", dtype=object)

        # Solid perimeter
        grid[0, :] = "wall"
        grid[-1, :] = "wall"
        grid[:, 0] = "wall"
        grid[:, -1] = "wall"

        s = self.square_size
        g = self.gap

        # Tile solid squares across the arena
        row = g + 1
        while row + s < H - 1:
            col = g + 1
            while col + s < W - 1:
                grid[row : row + s, col : col + s] = "wall"
                col += s + g
            row += s + g

        # Occupancy mask for later placement
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
