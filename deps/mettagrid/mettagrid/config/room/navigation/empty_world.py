"""
EmptyWorld
==========

Creates an empty rectangular arena—just a solid perimeter wall—with a sampled
number of altars and agents scattered inside.

Episode‑time sampling
---------------------
* **width, height**  ← uniform integers in `width_range`, `height_range`
* **altars_count**   ← uniform integer in `altars_count_range`  (default 14 – 40)
"""

from __future__ import annotations
from typing import Sequence
import numpy as np

from mettagrid.config.room.room import Room


class EmptyWorld(Room):
    STYLE_PARAMETERS = {"empty": {"hearts_count": 0}}

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        width_range: Sequence[int] = (60, 130),
        height_range: Sequence[int] = (60, 130),
        altars_count_range: Sequence[int] = (14, 40),
        agents: int | dict = 20,
        seed: int | None = None,
        border_object: str = "wall",
    ):
        super().__init__(border_object=border_object)
        rng = np.random.default_rng(seed)

        self.width = int(rng.integers(width_range[0], width_range[1] + 1))
        self.height = int(rng.integers(height_range[0], height_range[1] + 1))
        self.altars_count = int(
            rng.integers(altars_count_range[0], altars_count_range[1] + 1)
        )

        self._agents = agents
        self._rng = rng

    # ------------------------------------------------------------------ #
    def _build(self):
        grid = self._generate_grid()
        self._place_altars(grid)
        self._place_agents(grid)
        return grid

    # ------------------------------------------------------------------ #
    def _generate_grid(self) -> np.ndarray:
        """Return an object grid with a solid perimeter and empty interior."""
        H, W = self.height, self.width
        grid = np.full((H, W), "empty", dtype=object)

        # Solid outer wall
        grid[0, :] = grid[-1, :] = "wall"
        grid[:, 0] = grid[:, -1] = "wall"

        self._occ = np.zeros((H, W), dtype=bool)  # occupancy mask
        self._occ[grid == "wall"] = True
        return grid

    # ------------------------------------------------------------------ #
    def _place_altars(self, grid: np.ndarray):
        empties = list(zip(*np.where(~self._occ)))
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
        empties = np.flatnonzero(~self._occ)
        self._rng.shuffle(empties)
        for tag, idx in zip(tags, empties):
            pos = tuple(np.unravel_index(idx, grid.shape))
            grid[pos] = tag
            self._occ[pos] = True