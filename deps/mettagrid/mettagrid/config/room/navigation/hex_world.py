"""
HexWorld
========

Generates a *regular tiling* of flat‑top hexagons carved out of a square grid.

Key properties
--------------
* **hex_size ∈ [5 .. 10]** (inclusive, forced odd) – edge length of each hex.
* **gap** – fixed corridor width between neighbouring hexes (default 3).
* Every hex is *hollow* with a single doorway at the centre of its **east** side.
* A solid perimeter wall encloses the arena.
* Altars and agents are dropped uniformly in the empty cells.

Coordinate conventions assume (row, col) indexing.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np

from mettagrid.config.room.room import Room


# --------------------------------------------------------------------------- #
# Helper: draw a single flat‑top hexagon outline at (r0, c0) with side length s
# --------------------------------------------------------------------------- #
def _draw_hex(grid: np.ndarray, r0: int, c0: int, s: int):
    """Draws walls for a hexagon; leaves interior empty.  Adds an east‑side door."""
    H, W = grid.shape

    def safe_set(r: int, c: int):
        if 0 <= r < H and 0 <= c < W:
            grid[r, c] = "wall"

    # Upper half (rows 0 .. s‑1)
    for dr in range(s):
        row = r0 + dr
        span = s + dr
        col_start = c0 - dr
        col_end = col_start + span - 1
        safe_set(row, col_start)   # west edge
        safe_set(row, col_end)     # east edge

    # Lower half (rows s .. 2s‑1)
    for dr in range(s, 2 * s):
        row = r0 + dr
        span = s + (2 * s - dr - 1)
        col_start = c0 - (2 * s - dr - 1)
        col_end = col_start + span - 1
        safe_set(row, col_start)
        safe_set(row, col_end)

    # Door at centre of east side (upper middle row)
    door_r = r0 + s - 1
    door_c = c0 + s
    if 0 <= door_r < H and 0 <= door_c < W:
        grid[door_r, door_c] = "empty"


# --------------------------------------------------------------------------- #
class HexWorld(Room):
    """Room builder that tiles the arena with uniform hexagonal cells."""

    STYLE_PARAMETERS = {"hex_grid": {"hearts_count": 0}}

    # --------------------------------------------------------------- #
    def __init__(
        self,
        *,
        width_range: Sequence[int] = (60, 140),
        height_range: Sequence[int] = (60, 140),
        hex_size_range: Sequence[int] = (5, 10),
        gap: int = 3,
        altars_count: int = 50,
        agents: int | dict = 20,
        seed: int | None = None,
        border_object: str = "wall",
    ):
        super().__init__(border_object=border_object)
        rng = np.random.default_rng(seed)

        # Sample odd side length for centred doorway
        s = rng.integers(hex_size_range[0], hex_size_range[1] + 1)
        self.hex_size = s if s % 2 == 1 else (s + 1 if s < hex_size_range[1] else s - 1)
        self.gap = gap

        self.width = rng.integers(width_range[0], width_range[1] + 1)
        self.height = rng.integers(height_range[0], height_range[1] + 1)

        self.altars_count = altars_count
        self._agents = agents
        self._rng = rng
        self.set_size_labels(self.width, self.height, labels = ["hex_grid"])

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

        # Perimeter walls
        grid[0, :] = grid[-1, :] = "wall"
        grid[:, 0] = grid[:, -1] = "wall"

        s = self.hex_size
        w = 3 * s - 1          # bounding‑box width
        h = 2 * s              # bounding‑box height
        stride_x = w - (s - 1) # horizontal step between hex centres
        stride_y = h           # vertical step

        offset_y = 1 + s       # leave gap from top border
        offset_x = 1 + s

        row = offset_y
        parity = 0
        while row + s < H - 1:
            col = offset_x + parity * (stride_x // 2)
            while col + s < W - 1:
                _draw_hex(grid, row - (s - 1), col, s)
                col += stride_x
            row += stride_y
            parity ^= 1  # stagger every second row

        # Occupancy mask tracks filled cells for later placement
        self._occ = np.zeros((H, W), dtype=bool)
        self._occ[grid == "wall"] = True
        return grid

    # --------------------------------------------------------------- #
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
