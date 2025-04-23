"""
BoxWorld
========

Densely fills the map with rectangular *boxes*:

* Interior rectangle; one side is guaranteed ≥ 8 cells.
* Walls 1‑cell thick.
* A single *entrance* (one gap) sits at a randomly chosen corner.
* A single heart **altar** sits in the *diagonally opposite* corner.
* Boxes are stamped until 10 consecutive placement failures, then agents
  are spawned in the remaining open corridors.

Parameters (override in YAML `room:` block if desired):

    short_range  – inclusive (min, max) for interior short side
    elong_range  – inclusive (min_add, max_add) added to short → long side
"""

from typing import List, Optional, Tuple

import numpy as np
from mettagrid.config.room.room import Room


class BoxWorld(Room):
    STYLE_PARAMETERS = {"box_world": {"hearts_count": 0}}

    # ------------------------------------------------------------------ #
    # Init
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        width: int = 120,
        height: int = 120,
        agents: int | dict = 0,
        short_range: Tuple[int, int] = (4, 6),    # interior short side
        elong_range: Tuple[int, int] = (4, 8),    # + add to short → long
        seed: Optional[int] = 42,
        border_width: int = 2,
        border_object: str = "wall",
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._rng = np.random.default_rng(seed)
        self._width, self._height = np.random.randint(40,100), np.random.randint(40,100)
        self._agents = agents
        self._occ = np.zeros((height, width), dtype=bool)
        self.short_min, self.short_max = short_range
        self.elong_min, self.elong_max = elong_range

    # ------------------------------------------------------------------ #
    # Public build
    # ------------------------------------------------------------------ #
    def _build(self):
        grid = np.full((self._height, self._width), "empty", dtype=object)
        fails = 0
        while fails < 10:                               # pack until space gone
            if self._place_box(grid, clearance=1):
                fails = 0
            else:
                fails += 1
        return self._place_agents(grid)

    # ------------------------------------------------------------------ #
    # Box helpers
    # ------------------------------------------------------------------ #
    def _place_box(self, grid, clearance: int) -> bool:
        pattern = self._generate_box()
        return self._place_region(grid, pattern, clearance)

    def _generate_box(self) -> np.ndarray:
        # --- choose interior dimensions -------------------------------- #
        short = int(self._rng.integers(self.short_min, self.short_max + 1))
        long_add = int(self._rng.integers(self.elong_min, self.elong_max + 1))
        long_side = max(short + long_add, 8)            # ensure ≥ 8 somewhere

        horiz = self._rng.random() < 0.5
        h_int, w_int = (short, long_side) if horiz else (long_side, short)

        h, w = h_int + 2, w_int + 2                     # +2 for walls
        pat = np.full((h, w), "wall", dtype=object)
        pat[1:-1, 1:-1] = "empty"

        # pick entrance corner, altar opposite
        corners = [(1, 1), (1, w - 2), (h - 2, 1), (h - 2, w - 2)]
        entrance = corners[self._rng.integers(4)]
        altar = corners[(corners.index(entrance) + 2) % 4]
        er, ec = entrance

        pat[altar] = "altar"

        # carve a single gap adjacent to the entrance corner
        gaps = []
        if er == 1:          gaps.append((0, ec))          # top wall
        if er == h - 2:      gaps.append((h - 1, ec))      # bottom wall
        if ec == 1:          gaps.append((er, 0))          # left wall
        if ec == w - 2:      gaps.append((er, w - 1))      # right wall
        gap = gaps[self._rng.integers(len(gaps))]
        pat[gap] = "empty"
        return pat

    # ------------------------------------------------------------------ #
    # Agent placement
    # ------------------------------------------------------------------ #
    def _place_agents(self, grid):
        tags = (
            ["agent.agent"] * self._agents
            if isinstance(self._agents, int)
            else ["agent." + n for n, k in self._agents.items() for _ in range(k)]
        )
        for t in tags:
            pos = self._rand_empty()
            if pos:
                grid[pos] = t
                self._occ[pos] = True
        return grid

    # ------------------------------------------------------------------ #
    # Generic helpers (same as LabyrinthWorld)
    # ------------------------------------------------------------------ #
    def _place_region(self, grid, pat, clearance: int) -> bool:
        ph, pw = pat.shape
        for r, c in self._free_windows((ph + 2 * clearance, pw + 2 * clearance)):
            grid[r + clearance : r + clearance + ph,
                 c + clearance : c + clearance + pw] = pat
            self._occ[r : r + ph + 2 * clearance,
                      c : c + pw + 2 * clearance] = True
            return True
        return False

    def _free_windows(self, shape) -> List[Tuple[int, int]]:
        h, w = shape
        H, W = self._occ.shape
        if h > H or w > W:
            return []
        view = np.lib.stride_tricks.as_strided(
            self._occ, (H - h + 1, W - w + 1, h, w), self._occ.strides * 2
        )
        coords = np.argwhere(view.sum(axis=(2, 3)) == 0)
        self._rng.shuffle(coords)
        return [tuple(t) for t in coords]

    def _rand_empty(self):
        empties = np.flatnonzero(~self._occ)
        return None if empties.size == 0 else tuple(
            np.unravel_index(self._rng.integers(empties.size), self._occ.shape)
        )
