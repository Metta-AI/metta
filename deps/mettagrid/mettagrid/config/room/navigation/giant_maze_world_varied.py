"""
GiantMazeWorldVaried
====================

Variant of GiantMazeWorld with:

• Map dimensions sampled between 200 – 400 tiles.
• Main corridor width sampled from corridor_range.
• Fractal‑style side branches carved off main corridors.

Extra parameters
----------------
branch_prob       – chance an empty tile spawns a branch (0–1)
branch_scale      – branch width = int(main_width × branch_scale)
max_branch_length – upper bound on branch length (tiles)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mettagrid.config.room.giant_maze_world import GiantMazeWorld


def _sample_int(v: int | Sequence[int], rng: np.random.Generator) -> int:
    """Return v if int, else uniformly sample an int in [v[0], v[1]]."""
    if isinstance(v, Sequence):
        lo, hi = int(v[0]), int(v[1])
        return int(rng.integers(lo, hi + 1))
    return int(v)


def _sample_num(v, rng):
    """Sample float or int ranges transparently."""
    if isinstance(v, Sequence):
        lo, hi = v[0], v[1]
        if isinstance(lo, int) and isinstance(hi, int):
            return int(rng.integers(int(lo), int(hi) + 1))
        return float(rng.uniform(float(lo), float(hi)))
    return v


class GiantMazeWorldVaried(GiantMazeWorld):
    STYLE_PARAMETERS = {"giant_maze_varied": {"hearts_count": 0}}

    def __init__(
        self,
        width: int | Sequence[int] = (200, 400),
        height: int | Sequence[int] = (200, 400),
        corridor_range: Sequence[int] = (1, 10),
        wall_range: Sequence[int] = (1, 8),
        branch_prob: float = 0.3,
        branch_scale: float = 0.5,
        max_branch_length: int = 40,
        altars_count: int = 50,
        agents: int | dict = 0,
        seed: int | None = 42,
        border_width: int = 2,
        border_object: str = "wall",
    ):
        rng_ext = np.random.default_rng(seed)

        super().__init__(
            width=width,
            height=height,
            corridor_range=tuple(corridor_range),
            wall_range=tuple(wall_range),
            altars_count=altars_count,
            agents=agents,
            seed=int(rng_ext.integers(2**32)),
            border_width=border_width,
            border_object=border_object,
        )

        self.branch_prob = float(_sample_num(branch_prob, rng_ext))
        self.branch_scale = float(_sample_num(branch_scale, rng_ext))
        self.max_branch_length = int(_sample_num(max_branch_length, rng_ext))

    # ------------------------------------------------------------------
    # Maze generation with fractal branches
    # ------------------------------------------------------------------
    def _generate_maze(self) -> np.ndarray:
        grid = super()._generate_maze()  # base perfect maze

        branch_w = max(1, int(self.cw * self.branch_scale))
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        H, W = grid.shape
        rng = self._rng  # inherited RNG

        def wall(r, c):
            return 0 <= r < H and 0 <= c < W and grid[r, c] == "wall"

        for r in range(1, H - 1):
            for c in range(1, W - 1):
                if grid[r, c] != "empty" or rng.random() > self.branch_prob:
                    continue

                dr, dc = directions[rng.integers(4)]
                length = int(rng.integers(3, self.max_branch_length + 1))
                rr, cc = r + dr, c + dc

                for _ in range(length):
                    if not wall(rr, cc):
                        break

                    grid[rr, cc] = "empty"

                    # widen perpendicular
                    for off in range(1, branch_w):
                        if dr == 0:  # horizontal branch -> widen vertically
                            for s in (-1, 1):
                                rw = rr + s * off
                                if wall(rw, cc):
                                    grid[rw, cc] = "empty"
                        else:  # vertical branch -> widen horizontally
                            for s in (-1, 1):
                                cw = cc + s * off
                                if wall(rr, cw):
                                    grid[rr, cw] = "empty"

                    rr += dr
                    cc += dc

                    # 30 % chance to turn
                    if rng.random() < 0.3:
                        dr, dc = directions[rng.integers(4)]

        self._occ = np.zeros_like(grid, dtype=bool)
        return grid
