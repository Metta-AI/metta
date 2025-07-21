"""
Uniform-Grid Terrain
====================
A perfectly regular lattice:

  • Vertical walls every `cell_size` tiles
  • Horizontal walls every `cell_size` tiles
  • Each wall segment has a centred gap of width `gap_width` so agents can
    slip through.

Both `cell_size` (spaciousness) and `gap_width` are sweepable from YAML.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from omegaconf import DictConfig, ListConfig
from metta.mettagrid.room.room import Room


class UniformGridTerrain(Room):
    def __init__(
        self,
        width: int = 160,
        height: int = 160,
        agents: int = 1,
        objects: DictConfig | dict | None = None,
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: str = "wall",
        #
        cell_size: int = 8,
        gap_width: int = 2,
    ) -> None:
        super().__init__(
            border_width=border_width,
            border_object=border_object,
            labels=["uniform_grid"],
        )
        self.set_size_labels(width, height)

        self._H, self._W = height, width
        self._rng = np.random.default_rng(seed)
        self._agents = agents if isinstance(agents, int) else 1
        self._objects = objects or {}

        # ------------------------------------------------------------------
        # Resolve sweep ranges (lists / tuples / ListConfig) into integers
        # ------------------------------------------------------------------
        def _sample(val, minimum: int) -> int:
            if isinstance(val, (list, tuple, ListConfig)):
                lo, hi = int(val[0]), int(val[1])
                if hi < lo:
                    lo, hi = hi, lo
                return max(minimum, int(self._rng.integers(lo, hi + 1)))
            return max(minimum, int(val))

        self._cell = _sample(cell_size, 2)
        self._gap  = _sample(gap_width, 1)

        # Ensure the doorway gap is always narrower than the cell spacing
        self._gap = min(self._gap, self._cell - 1)
        self._occ = np.zeros((height, width), dtype=bool)

    # ------------------------------------------------------------------ #
    # Room construction                                                  #
    # ------------------------------------------------------------------ #
    def _build(self) -> np.ndarray:
        grid = np.full((self._H, self._W), "empty", dtype=object)

        # 0 ─ border walls
        grid[0, :] = grid[-1, :] = "wall"
        grid[:, 0] = grid[:, -1] = "wall"
        self._occ[0, :], self._occ[-1, :] = True, True
        self._occ[:, 0], self._occ[:, -1] = True, True

        # 1 ─ vertical walls + gaps
        for c in range(self._cell, self._W - 1, self._cell):
            grid[:, c] = "wall"
            self._occ[:, c] = True
            for r0 in range(self._cell, self._H - 1, self._cell):
                gap_start = max(1, min(r0 - self._gap // 2, self._H - 1 - self._gap))
                grid[gap_start : gap_start + self._gap, c] = "empty"
                self._occ[gap_start : gap_start + self._gap, c] = False

        # 2 ─ horizontal walls + gaps
        for r in range(self._cell, self._H - 1, self._cell):
            grid[r, :] = "wall"
            self._occ[r, :] = True
            for c0 in range(self._cell, self._W - 1, self._cell):
                gap_start = max(1, min(c0 - self._gap // 2, self._W - 1 - self._gap))
                grid[r, gap_start : gap_start + self._gap] = "empty"
                self._occ[r, gap_start : gap_start + self._gap] = False

        # 3 ─ place agents
        for _ in range(self._agents):
            pos = self._pick_empty()
            if pos is None:
                break
            r, c = pos
            grid[r, c] = "agent.agent"
            self._occ[r, c] = True

        # 4 ─ user objects
        for name, cnt in (self._objects or {}).items():
            for _ in range(int(cnt)):
                pos = self._pick_empty()
                if pos is None:
                    break
                r, c = pos
                grid[r, c] = name
                self._occ[r, c] = True

        return grid

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _pick_empty(self) -> Optional[Tuple[int, int]]:
        empty_flat = np.flatnonzero(~self._occ)
        if empty_flat.size == 0:
            return None
        idx = self._rng.integers(empty_flat.size)
        return np.unravel_index(idx, self._occ.shape)
