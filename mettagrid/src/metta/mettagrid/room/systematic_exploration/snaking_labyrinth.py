"""
Snaking-Labyrinth Terrain
=========================
* Wide passages (≥2) with thin or thick walls (1-2).
* Altars placed at dead-ends first.
* NEW: Random single-tile wall blocks sprinkled along existing walls
  to add clutter, controlled by `inner_block_prob`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room


class SnakingLabyrinthTerrain(Room):
    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        width: int = 120,
        height: int = 120,
        agents: int = 4,
        objects: DictConfig | Dict[str, int] | None = None,
        seed: Optional[int] = None,
        *,
        corridor_width: int = 2,
        wall_thickness: int = 2,
        altar_count: int = 25,
        inner_block_prob: float = 0.05,  # ← sprinkle probability
    ) -> None:
        if corridor_width < 2:
            raise ValueError("corridor_width must be at least 2")
        if wall_thickness not in (1, 2):
            raise ValueError("wall_thickness must be 1 or 2")
        if not 0 <= inner_block_prob <= 1:
            raise ValueError("inner_block_prob must be in [0,1]")

        super().__init__(border_width=0, border_object="wall", labels=["snaking_labyrinth"])
        self.set_size_labels(width, height)

        self._H, self._W = height, width
        self._rng = np.random.default_rng(seed)

        self._agents = int(agents)
        self._altar_count = int(altar_count)
        self._objects = {} if objects is None else dict(objects)

        self._cw = corridor_width
        self._wt = wall_thickness
        self._p_block = float(inner_block_prob)

        self._occ: np.ndarray  # set in _build

    # ------------------------------------------------------------------ #
    # Room API                                                            #
    # ------------------------------------------------------------------ #
    def _build(self) -> np.ndarray:
        coarse = self._generate_maze()
        grid = self._scale_and_carve(coarse)
        self._sprinkle_inner_blocks(grid)  # ← new step
        self._scatter_entities(grid)
        return grid

    # ------------------------------------------------------------------ #
    # 1. Generate DFS maze                                                #
    # ------------------------------------------------------------------ #
    def _generate_maze(self) -> np.ndarray:
        step = self._cw + self._wt
        rows = ((self._H - self._wt) // step) | 1
        cols = ((self._W - self._wt) // step) | 1

        maze = np.ones((rows, cols), dtype=np.int8)
        stack = [(1, 1)]
        maze[1, 1] = 0
        dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]

        while stack:
            r, c = stack[-1]
            nbrs = [
                (r + dr, c + dc)
                for dr, dc in dirs
                if 0 < r + dr < rows - 1 and 0 < c + dc < cols - 1 and maze[r + dr, c + dc]
            ]
            if nbrs:
                nr, nc = nbrs[self._rng.integers(len(nbrs))]
                maze[(r + nr) // 2, (c + nc) // 2] = 0
                maze[nr, nc] = 0
                stack.append((nr, nc))
            else:
                stack.pop()
        return maze

    # ------------------------------------------------------------------ #
    # 2. Scale coarse maze                                                #
    # ------------------------------------------------------------------ #
    def _scale_and_carve(self, maze: np.ndarray) -> np.ndarray:
        rows, cols = maze.shape
        out_h = rows * self._cw + (rows + 1) * self._wt
        out_w = cols * self._cw + (cols + 1) * self._wt
        grid = np.full((out_h, out_w), "wall", dtype=object)

        def cell_top(r):
            return r * (self._cw + self._wt) + self._wt

        def cell_left(c):
            return c * (self._cw + self._wt) + self._wt

        for r in range(rows):
            for c in range(cols):
                if maze[r, c] == 0:
                    rt, cl = cell_top(r), cell_left(c)
                    grid[rt : rt + self._cw, cl : cl + self._cw] = "empty"
                    if c + 1 < cols and maze[r, c + 1] == 0:
                        cc = cl + self._cw
                        grid[rt : rt + self._cw, cc : cc + self._wt] = "empty"
                    if r + 1 < rows and maze[r + 1, c] == 0:
                        rr = rt + self._cw
                        grid[rr : rr + self._wt, cl : cl + self._cw] = "empty"

        full = np.full((self._H, self._W), "wall", dtype=object)
        full[:out_h, :out_w] = grid[: self._H, : self._W]
        self._occ = full == "wall"
        return full

    # ------------------------------------------------------------------ #
    # NEW 2a. Sprinkle single wall blocks alongside corridors             #
    # ------------------------------------------------------------------ #
    def _sprinkle_inner_blocks(self, grid: np.ndarray) -> None:
        if self._p_block <= 0:
            return
        candidates: List[Tuple[int, int]] = []
        for r in range(1, self._H - 1):
            for c in range(1, self._W - 1):
                if grid[r, c] != "empty":
                    continue
                if (
                    grid[r - 1, c] == "wall"
                    or grid[r + 1, c] == "wall"
                    or grid[r, c - 1] == "wall"
                    or grid[r, c + 1] == "wall"
                ):
                    candidates.append((r, c))
        self._rng.shuffle(candidates)
        k = int(self._p_block * len(candidates))
        for r, c in candidates[:k]:
            grid[r, c] = "wall"
            self._occ[r, c] = True

    # ------------------------------------------------------------------ #
    # 3. Scatter entities                                                 #
    # ------------------------------------------------------------------ #
    def _scatter_entities(self, grid: np.ndarray) -> None:
        empty = ~self._occ

        # altars at dead-ends
        dead_ends = [
            (r, c)
            for r in range(1, self._H - 1)
            for c in range(1, self._W - 1)
            if empty[r, c] and (empty[r - 1, c] + empty[r + 1, c] + empty[r, c - 1] + empty[r, c + 1]) == 1
        ]
        self._rng.shuffle(dead_ends)
        for r, c in dead_ends[: self._altar_count]:
            grid[r, c] = "altar"
            empty[r, c] = False

        # agents
        empties = np.flatnonzero(empty)
        for _ in range(self._agents):
            if not len(empties):
                break
            idx = self._rng.integers(len(empties))
            r, c = np.unravel_index(empties[idx], empty.shape)
            grid[r, c] = "agent.agent"
            empty[r, c] = False
            empties = np.flatnonzero(empty)

        # extra objects
        for name, cnt in self._objects.items():
            for _ in range(int(cnt)):
                if not len(empties):
                    return
                idx = self._rng.integers(len(empties))
                r, c = np.unravel_index(empties[idx], empty.shape)
                grid[r, c] = name
                empty[r, c] = False
                empties = np.flatnonzero(empty)
