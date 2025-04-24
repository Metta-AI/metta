"""
GiantMazeWorld (v3, continuous)
===============================

Builds one perfect maze, then up‑scales it so corridors and walls have
variable thickness **while preserving full connectivity**.

User‑tunable ranges (sampled each episode):

* corridor_range  – inclusive [min,max] corridor width (≥ 1)
* wall_range      – inclusive [min,max] wall thickness (≥ 1)
* altars_count    – number of heart altars sprinkled on corridor tiles
"""

from typing import List, Optional, Tuple

import numpy as np

from mettagrid.config.room.room import Room


class GiantMazeWorld(Room):
    STYLE_PARAMETERS = {"giant_maze": {"hearts_count": 0}}

    # --------------------------------------------------------------- #
    def __init__(
        self,
        width: Tuple[int, int] = (80, 200),
        height: Tuple[int, int] = (80, 200),
        corridor_range: Tuple[int, int] = (1, 10),
        wall_range: Tuple[int, int] = (1, 5),
        altars_count: int = 50,
        agents: int | dict = 0,
        seed: Optional[int] = 42,
        border_width: int = 2,
        border_object: str = "wall",
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._rng = np.random.default_rng(seed)

        width: int = np.random.randint(width[0], width[1]) | 1
        height: int = np.random.randint(height[0], height[1]) | 1

        # force odd dims (good for maze generation)
        self.base_w = width
        self.base_h = height

        self.corridor_range = corridor_range
        self.wall_range = wall_range
        self.altars_count = altars_count
        self._agents = agents
        self.set_size_labels(width, height, labels = ["giant_maze"])

    # --------------------------------------------------------------- #
    # Public builder
    # --------------------------------------------------------------- #
    def _build(self):
        # sample widths
        self.cw = int(self._rng.integers(self.corridor_range[0], self.corridor_range[1] + 1))
        self.wt = int(self._rng.integers(self.wall_range[0], self.wall_range[1] + 1))

        grid = self._generate_maze()
        self._place_altars(grid)
        self._place_agents(grid)
        return grid

    # --------------------------------------------------------------- #
    # Maze generation with connectivity‑preserving upscale
    # --------------------------------------------------------------- #
    def _generate_maze(self) -> np.ndarray:
        cw, wt = self.cw, self.wt
        cell = cw + wt
        # coarse grid size (odd)
        small_h = ((self.base_h - 1) // cell) | 1
        small_w = ((self.base_w - 1) // cell) | 1

        # perfect maze on coarse grid (True=corridor)
        coarse = np.full((small_h, small_w), False, dtype=bool)
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        stack = [(1, 1)]
        coarse[1, 1] = True
        last_dir = None
        rng = self._rng
        while stack:
            r, c = stack[-1]
            opts = [
                (dr, dc)
                for dr, dc in dirs
                if 0 < r + dr * 2 < small_h - 1 and 0 < c + dc * 2 < small_w - 1 and not coarse[r + dr * 2, c + dc * 2]
            ]
            if opts:
                if last_dir in opts and rng.random() > 0.3:
                    dr, dc = last_dir
                else:
                    dr, dc = opts[rng.integers(len(opts))]
                coarse[r + dr, c + dc] = True
                coarse[r + dr * 2, c + dc * 2] = True
                stack.append((r + dr * 2, c + dc * 2))
                last_dir = (dr, dc)
            else:
                stack.pop()
                last_dir = None

        # upscale to full resolution with passages through walls
        big_h, big_w = small_h * cell + 1, small_w * cell + 1
        grid = np.full((big_h, big_w), "wall", dtype=object)

        # carve blocks
        for sr in range(small_h):
            for sc in range(small_w):
                if coarse[sr, sc]:
                    r0, c0 = sr * cell + wt, sc * cell + wt
                    grid[r0 : r0 + cw, c0 : c0 + cw] = "empty"

        # carve connectors through walls
        for sr in range(small_h):
            for sc in range(small_w):
                if not coarse[sr, sc]:
                    continue
                # right neighbour
                if sc + 1 < small_w and coarse[sr, sc + 1]:
                    r0 = sr * cell + wt
                    c_bridge = (sc + 1) * cell
                    grid[r0 : r0 + cw, c_bridge : c_bridge + wt] = "empty"
                # bottom neighbour
                if sr + 1 < small_h and coarse[sr + 1, sc]:
                    c0 = sc * cell + wt
                    r_bridge = (sr + 1) * cell
                    grid[r_bridge : r_bridge + wt, c0 : c0 + cw] = "empty"

        # occupancy mask
        self._height, self._width = big_h, big_w
        self._occ = np.zeros((big_h, big_w), dtype=bool)
        return grid

    # --------------------------------------------------------------- #
    # Altars & agents
    # --------------------------------------------------------------- #
    def _place_altars(self, grid: np.ndarray):
        empties = list(zip(*np.where(grid == "empty"), strict=False))
        self._rng.shuffle(empties)
        for pos in empties[: self.altars_count]:
            grid[pos] = "altar"
            self._occ[pos] = True

    def _place_agents(self, grid: np.ndarray):
        tags: List[str] = (
            ["agent.agent"] * self._agents
            if isinstance(self._agents, int)
            else ["agent." + t for t, n in self._agents.items() for _ in range(n)]
        )
        empties = np.flatnonzero((grid == "empty") & (~self._occ))
        self._rng.shuffle(empties)
        for tag, idx in zip(tags, empties, strict=False):
            pos = tuple(np.unravel_index(idx, grid.shape))
            grid[pos] = tag
            self._occ[pos] = True
