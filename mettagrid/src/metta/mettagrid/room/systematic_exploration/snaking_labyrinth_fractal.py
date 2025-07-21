"""
Snaking‑Labyrinth *Fractal* Terrain
===================================
A two‑scale maze:

1. **Outer maze** – corridors 3 cells wide, walls 2 cells thick.
2. **Inner spurs** – 1‑cell‑wide tunnels grown from outer dead‑ends,
   each ending with an altar.

Agents will see an altar through a 1‑tile throat, must back‑track along the
wide corridor, then squeeze down a narrow spur to collect it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room


class SnakingLabyrinthFractalTerrain(Room):
    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        width: int = 240,
        height: int = 240,
        agents: int = 4,
        objects: DictConfig | Dict[str, int] | None = None,
        seed: Optional[int] = None,
        #
        outer_corridor: int = 3,  # ≥2
        outer_wall: int = 2,
        spur_length_range: Tuple[int, int] = (8, 18),
        altar_count: int = 50,
    ) -> None:
        super().__init__(border_width=0, border_object="wall", labels=["snaking_labyrinth_fractal"])
        self.set_size_labels(width, height)

        self._H, self._W = height, width
        self._rng = np.random.default_rng(seed)

        self._agents = int(agents)
        self._altar_target = int(altar_count)
        self._objects = {} if objects is None else dict(objects)

        # geometry parameters
        self._cw = max(2, outer_corridor)
        self._wt = max(1, outer_wall)
        self._spur_rng = spur_length_range

        self._occ: np.ndarray  # set in _build

    # ------------------------------------------------------------------ #
    # Room API                                                            #
    # ------------------------------------------------------------------ #
    def _build(self) -> np.ndarray:
        # 1 ─ outer wide maze
        outer_coarse = self._dfs_maze_coarse()
        grid = self._scale_maze(outer_coarse)

        # 2 ─ narrow spurs + altars
        self._carve_spurs_with_altars(grid)

        # 3 ─ agents and optional extras
        self._scatter_agents_and_objects(grid)

        return grid

    # ------------------------------------------------------------------ #
    # Outer maze generation                                               #
    # ------------------------------------------------------------------ #
    def _dfs_maze_coarse(self) -> np.ndarray:
        step = self._cw + self._wt
        rows = ((self._H - self._wt) // step) | 1  # odd
        cols = ((self._W - self._wt) // step) | 1

        maze = np.ones((rows, cols), np.int8)
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
    # Scale coarse maze to wide corridors                                #
    # ------------------------------------------------------------------ #
    def _scale_maze(self, maze: np.ndarray) -> np.ndarray:
        rows, cols = maze.shape
        out_h = rows * self._cw + (rows + 1) * self._wt
        out_w = cols * self._cw + (cols + 1) * self._wt
        grid = np.full((out_h, out_w), "wall", dtype=object)

        def get_r(r):
            return r * (self._cw + self._wt) + self._wt

        def get_c(c):
            return c * (self._cw + self._wt) + self._wt

        # carve passages
        for r in range(rows):
            for c in range(cols):
                if maze[r, c] == 0:
                    rt, cl = get_r(r), get_c(c)
                    grid[rt : rt + self._cw, cl : cl + self._cw] = "empty"
                    # right connector
                    if c + 1 < cols and maze[r, c + 1] == 0:
                        cc = cl + self._cw
                        grid[rt : rt + self._cw, cc : cc + self._wt] = "empty"
                    # down connector
                    if r + 1 < rows and maze[r + 1, c] == 0:
                        rr = rt + self._cw
                        grid[rr : rr + self._wt, cl : cl + self._cw] = "empty"

        # crop/pad to requested H×W
        full = np.full((self._H, self._W), "wall", dtype=object)
        copy_h = min(out_h, self._H)
        copy_w = min(out_w, self._W)
        full[:copy_h, :copy_w] = grid[:copy_h, :copy_w]
        self._occ = full == "wall"
        return full

    # ------------------------------------------------------------------ #
    # Spur carving & altar placement                                     #
    # ------------------------------------------------------------------ #
    def _carve_spurs_with_altars(self, grid: np.ndarray) -> None:
        empty = ~self._occ
        dead_ends: List[Tuple[int, int]] = []
        for r in range(1, self._H - 1):
            for c in range(1, self._W - 1):
                if empty[r, c]:
                    deg = empty[r - 1, c] + empty[r + 1, c] + empty[r, c - 1] + empty[r, c + 1]
                    if deg == 1:
                        dead_ends.append((r, c))

        self._rng.shuffle(dead_ends)
        altars_placed = 0
        for r, c in dead_ends:
            if altars_placed >= self._altar_target:
                break
            # direction pointing *away* from corridor
            if empty[r - 1, c]:
                dr, dc = 1, 0
            elif empty[r + 1, c]:
                dr, dc = -1, 0
            elif empty[r, c - 1]:
                dr, dc = 0, 1
            else:
                dr, dc = 0, -1

            length = int(self._rng.integers(*self._spur_rng))
            rr, cc = r + dr, c + dc
            carved = 0
            while 0 < rr < self._H - 1 and 0 < cc < self._W - 1 and self._occ[rr, cc] and carved < length:
                grid[rr, cc] = "empty"
                self._occ[rr, cc] = False
                empty[rr, cc] = True
                rr += dr
                cc += dc
                carved += 1

            # place altar at tip if carved at least 2 cells
            if carved >= 2 and grid[rr - dr, cc - dc] == "empty":
                altar_r, altar_c = rr - dr, cc - dc
                grid[altar_r, altar_c] = "altar"
                altars_placed += 1
                empty[altar_r, altar_c] = False

    # ------------------------------------------------------------------ #
    # Agents and extra objects                                           #
    # ------------------------------------------------------------------ #
    def _scatter_agents_and_objects(self, grid: np.ndarray) -> None:
        empty = ~(self._occ)

        # agents
        empties_flat = np.flatnonzero(empty)
        for _ in range(self._agents):
            if not empties_flat.size:
                break
            idx = self._rng.integers(empties_flat.size)
            r, c = np.unravel_index(empties_flat[idx], empty.shape)
            grid[r, c] = "agent.agent"
            empty[r, c] = False
            empties_flat = np.flatnonzero(empty)

        # user‑specified extra objects (rare)
        for name, cnt in self._objects.items():
            for _ in range(int(cnt)):
                if not empties_flat.size:
                    return
                idx = self._rng.integers(empties_flat.size)
                r, c = np.unravel_index(empties_flat[idx], empty.shape)
                grid[r, c] = name
                empty[r, c] = False
                empties_flat = np.flatnonzero(empty)
