"""
HomogenousTile
==============

Tiles the arena with identical rooms separated by *single‑cell walls* and
one‑cell corridors.  Each episode:

* Samples **tile_width , tile_height** ∈ [5 .. 10]
* Samples **arena width , height**    ∈ [60 .. 130]
* Samples **altars_count**            ∈ [14 .. 40]
* Picks a **shape** for every tile interior from
  {rectangle, cross, hollow, diagonal}.

Doors are carved at the centre of every interior wall so the agent can move
freely between tiles.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
import random

from mettagrid.config.room.room import Room


class HomogenousTile(Room):
    STYLE_PARAMETERS = {"homogenous_tile": {"hearts_count": 0}}

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        width_range: Sequence[int] = (60, 130),
        height_range: Sequence[int] = (60, 130),
        tile_width_range: Sequence[int] = (5, 10),
        tile_height_range: Sequence[int] = (5, 10),
        altars_count_range: Sequence[int] = (14, 40),
        agents: int | dict = 20,
        seed: int | None = None,
        border_object: str = "wall",
    ):
        super().__init__(border_object=border_object, labels = ["homogenous_tile"])
        rng = np.random.default_rng(seed)

        self.width  = int(rng.integers(width_range[0],  width_range[1]  + 1))
        self.height = int(rng.integers(height_range[0], height_range[1] + 1))

        self.tile_w = int(rng.integers(tile_width_range[0],  tile_width_range[1]  + 1))
        self.tile_h = int(rng.integers(tile_height_range[0], tile_height_range[1] + 1))

        self.altars_count = int(
            rng.integers(altars_count_range[0], altars_count_range[1] + 1)
        )

        self.shape = random.choice(["rectangle", "cross", "hollow", "diagonal"])
        self.gap   = 1  # single‑cell corridor
        self._agents = agents
        self._rng = rng
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

        # Perimeter
        grid[0, :] = grid[-1, :] = "wall"
        grid[:, 0] = grid[:, -1] = "wall"

        tw, th, gap = self.tile_w, self.tile_h, self.gap
        stride_x = tw + gap + 1   # tile + corridor + wall
        stride_y = th + gap + 1

        # Draw lattice walls
        for c in range(0, W, stride_x):
            grid[:, c] = "wall"
        for r in range(0, H, stride_y):
            grid[r, :] = "wall"

        # Carve interior patterns
        for r0 in range(0, H, stride_y):
            for c0 in range(0, W, stride_x):
                r_start = r0 + 1         # first cell inside north wall
                c_start = c0 + 1
                r_end = min(r_start + th, H - 1)
                c_end = min(c_start + tw, W - 1)

                if r_end >= H - 1 or c_end >= W - 1:
                    continue  # skip partial tile at border

                # Fill interior with walls, then carve shape
                grid[r_start:r_end, c_start:c_end] = "wall"
                mid_r = r_start + th // 2
                mid_c = c_start + tw // 2

                if self.shape == "rectangle":
                    grid[r_start:r_end, c_start:c_end] = "empty"

                elif self.shape == "cross":
                    grid[r_start:r_end, mid_c] = "empty"
                    grid[mid_r, c_start:c_end] = "empty"

                elif self.shape == "hollow":
                    grid[r_start,   c_start:c_end] = "empty"
                    grid[r_end-1,   c_start:c_end] = "empty"
                    grid[r_start:r_end, c_start]   = "empty"
                    grid[r_start:r_end, c_end-1]   = "empty"

                elif self.shape == "diagonal":
                    for d in range(min(th, tw)):
                        rr, cc = r_start + d, c_start + d
                        if rr < r_end and cc < c_end:
                            grid[rr, cc] = "empty"

        # Carve doors (centred) in vertical walls
        for c in range(stride_x, W - 1, stride_x):
            for r0 in range(0, H, stride_y):
                door_r = r0 + 1 + th // 2
                if door_r < H - 1 and grid[door_r, c] == "wall":
                    grid[door_r, c] = "empty"

        # Carve doors (centred) in horizontal walls
        for r in range(stride_y, H - 1, stride_y):
            for c0 in range(0, W, stride_x):
                door_c = c0 + 1 + tw // 2
                if door_c < W - 1 and grid[r, door_c] == "wall":
                    grid[r, door_c] = "empty"

        self._occ = np.zeros((H, W), dtype=bool)
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
