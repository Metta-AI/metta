"""
Fixed‑Sweep Trap Terrain
========================
Breaks a boustrophedon (lawn‑mower) robot by combining:

• **Jagged perimeter.**  Random saw‑tooth indentations along all four
  edges create non‑rectilinear boundaries; straight sweeps leave wedge‑shaped
  gaps or collide with the walls early.

• **Interior islands.**  Rectangular blocks at random positions desynchronise
  the row/turn cadence and produce further uncovered stripes.

Everything is configurable from YAML.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room


class FixedSweepTerrain(Room):
    """Terrain that defeats fixed lawn‑mower sweep patterns."""

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        width: int = 120,
        height: int = 100,
        agents: int = 1,
        objects: DictConfig | dict | None = None,
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: str = "wall",
        #
        indent_count: Tuple[int, int] = (10, 24),  # total indentations per side
        indent_depth: Tuple[int, int] = (3, 10),
        indent_width: Tuple[int, int] = (3, 7),
        #
        island_count: Tuple[int, int] = (25, 45),
        island_size: Tuple[int, int] = (4, 12),  # rectangular islands
        #
        occupancy_threshold: float = 0.60,
    ) -> None:
        super().__init__(
            border_width=border_width,
            border_object=border_object,
            labels=["fixed_sweep"],
        )
        self.set_size_labels(width, height)

        self._H, self._W = height, width
        self._rng = np.random.default_rng(seed)

        self._agents = agents if isinstance(agents, int) else 1
        self._objects = objects or {}

        # parameters
        self._indent_cnt = indent_count
        self._indent_depth = indent_depth
        self._indent_width = indent_width

        self._island_cnt = island_count
        self._island_sz = island_size

        self._occ_thr = occupancy_threshold
        self._occ = np.zeros((height, width), dtype=bool)

    # ------------------------------------------------------------------ #
    # Room API                                                            #
    # ------------------------------------------------------------------ #
    def _build(self) -> np.ndarray:
        grid = np.full((self._H, self._W), "empty", dtype=object)

        # 1 ─ build irregular perimeter
        self._carve_jagged_perimeter(grid)

        # 2 ─ scatter interior islands
        self._scatter_islands(grid)

        # 3 ─ agents & objects
        self._scatter_agents_and_objects(grid)

        return grid

    # ------------------------------------------------------------------ #
    # Jagged perimeter generation                                         #
    # ------------------------------------------------------------------ #
    def _carve_jagged_perimeter(self, grid: np.ndarray) -> None:
        # First lay a continuous 1‑cell border.
        grid[0, :] = "wall"
        grid[-1, :] = "wall"
        grid[:, 0] = "wall"
        grid[:, -1] = "wall"
        self._occ[0, :] = True
        self._occ[-1, :] = True
        self._occ[:, 0] = True
        self._occ[:, -1] = True

        total_per_side = self._rng.integers(*self._indent_cnt)

        for side in ("top", "bottom", "left", "right"):
            for _ in range(total_per_side):
                depth = int(self._rng.integers(*self._indent_depth))
                width = int(self._rng.integers(*self._indent_width))
                if side in ("top", "bottom"):
                    max_start = self._W - width - 2
                    if max_start <= 1:
                        continue
                    c0 = self._rng.integers(1, max_start)
                    rows = range(depth) if side == "top" else range(self._H - depth, self._H)
                    grid[np.array(rows)[:, None], c0 : c0 + width] = "wall"
                    self._occ[np.array(rows)[:, None], c0 : c0 + width] = True
                else:  # left or right
                    max_start = self._H - width - 2
                    if max_start <= 1:
                        continue
                    r0 = self._rng.integers(1, max_start)
                    cols = range(depth) if side == "left" else range(self._W - depth, self._W)
                    grid[r0 : r0 + width, np.array(cols)] = "wall"
                    self._occ[r0 : r0 + width, np.array(cols)] = True

    # ------------------------------------------------------------------ #
    # Island generation                                                  #
    # ------------------------------------------------------------------ #
    def _scatter_islands(self, grid: np.ndarray) -> None:
        n_islands = self._rng.integers(*self._island_cnt)
        for _ in range(n_islands):
            if self._occ.mean() >= self._occ_thr:
                break
            h = int(self._rng.integers(*self._island_sz))
            w = int(self._rng.integers(*self._island_sz))
            pattern = np.full((h, w), "wall", dtype=object)
            self._try_place(grid, pattern, clearance=1)

    # ------------------------------------------------------------------ #
    # Placement helpers                                                  #
    # ------------------------------------------------------------------ #
    def _try_place(self, grid: np.ndarray, pattern: np.ndarray, *, clearance: int) -> bool:
        ph, pw = pattern.shape
        candidates = self._find_candidates((ph, pw), clearance)
        if not candidates:
            return False
        r, c = candidates[self._rng.integers(len(candidates))]
        grid[r : r + ph, c : c + pw] = pattern
        self._occ[r - clearance : r + ph + clearance, c - clearance : c + pw + clearance] = True
        return True

    def _find_candidates(self, shape: Tuple[int, int], clearance: int) -> List[Tuple[int, int]]:
        ph, pw = shape
        H, W = self._occ.shape
        if ph + 2 * clearance > H or pw + 2 * clearance > W:
            return []
        occ = self._occ
        ok: list[Tuple[int, int]] = []
        for r in range(clearance, H - ph - clearance + 1):
            for c in range(clearance, W - pw - clearance + 1):
                sub = occ[r - clearance : r + ph + clearance, c - clearance : c + pw + clearance]
                if not sub.any():
                    ok.append((r, c))
        return ok

    # ------------------------------------------------------------------ #
    # Agents & objects                                                   #
    # ------------------------------------------------------------------ #
    def _scatter_agents_and_objects(self, grid: np.ndarray) -> None:
        occ = self._occ | (grid != "empty")

        for _ in range(self._agents):
            pos = self._pick_empty(occ)
            if pos is None:
                break
            r, c = pos
            grid[r, c] = "agent.agent"
            occ[r, c] = True

        for name, cnt in self._objects.items():
            for _ in range(cnt):
                pos = self._pick_empty(occ)
                if pos is None:
                    break
                r, c = pos
                grid[r, c] = name
                occ[r, c] = True

    def _pick_empty(self, occ: np.ndarray) -> Optional[Tuple[int, int]]:
        flat = np.flatnonzero(~occ)
        if flat.size == 0:
            return None
        idx = self._rng.integers(flat.size)
        return np.unravel_index(flat[idx], occ.shape)
