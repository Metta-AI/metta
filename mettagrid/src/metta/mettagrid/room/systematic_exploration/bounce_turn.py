"""
Bounce-and-Turn Trap Terrain
===========================
Designed to confine a memory-less 'bug 0' agent that:

    – drives straight until it meets a wall,
    – then glides along the obstacle edge
      until it can resume its original heading.

Key features
------------
* **Concave pockets (C-shapes).**  Once the agent slides inside,
  the interior rim keeps it orbiting forever.
* **Venus-fly-trap funnels.** Two obstacles form a narrow throat
  (1-cell gap) that admits entry but closes off deeper inside.
* **Random extra corner block.** One inner-corner cell becomes a wall
  (if empty) to break symmetry and add a tiny extra obstacle.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room


class BounceTurnTerrain(Room):
    """Terrain that breaks bug-0 style navigation."""

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        width: int = 100,
        height: int = 100,
        agents: int = 1,
        objects: DictConfig | dict | None = None,
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: str = "wall",
        #
        pocket_count: Tuple[int, int] = (15, 30),
        pocket_size: Tuple[int, int] = (6, 15),  # inner cavity side length
        trap_count: Tuple[int, int] = (15, 30),
        trap_depth: Tuple[int, int] = (8, 20),
        trap_width: Tuple[int, int] = (5, 11),  # outer width (odd ≥ 5)
        occupancy_threshold: float = 0.55,
    ) -> None:
        super().__init__(
            border_width=border_width,
            border_object=border_object,
            labels=["bounce_turn"],
        )
        self.set_size_labels(width, height)

        self._H, self._W = height, width
        self._rng = np.random.default_rng(seed)

        self._agents = agents if isinstance(agents, int) else 1
        self._objects = objects or {}

        # parameters
        self._pocket_cnt = pocket_count
        self._pocket_size = pocket_size
        self._trap_cnt = trap_count
        self._trap_depth = trap_depth
        self._trap_width = trap_width
        self._occ_thr = occupancy_threshold

        self._occ = np.zeros((height, width), dtype=bool)

    # ------------------------------------------------------------------ #
    # Room API                                                            #
    # ------------------------------------------------------------------ #
    def _build(self) -> np.ndarray:
        grid = np.full((self._H, self._W), "empty", dtype=object)

        # 1 ─ concave pockets
        self._scatter_pockets(grid)

        # 2 ─ Venus-fly-traps
        self._scatter_traps(grid)

        # 3 ─ agents & objects
        self._scatter_agents_and_objects(grid)

        # 4 ─ extra random corner block (if that corner is still empty)
        corners = [
            (1, 1),
            (1, self._W - 2),
            (self._H - 2, 1),
            (self._H - 2, self._W - 2),
        ]
        self._rng.shuffle(corners)
        for r, c in corners:
            if grid[r, c] == "empty":
                grid[r, c] = "wall"
                break

        return grid

    # ------------------------------------------------------------------ #
    # Concave pockets                                                    #
    # ------------------------------------------------------------------ #
    def _scatter_pockets(self, grid: np.ndarray) -> None:
        n_pockets = self._rng.integers(*self._pocket_cnt)
        for _ in range(n_pockets):
            if self._occ.mean() >= self._occ_thr:
                break
            size = int(self._rng.integers(*self._pocket_size))  # cavity side
            pattern = self._make_c_pocket(size)
            self._try_place(grid, pattern, clearance=1)

    def _make_c_pocket(self, cavity: int) -> np.ndarray:
        """
        Build a 'C' shaped pocket; outer walls 1-cell thick.
        Randomly rotate 0°, 90°, 180°, 270° to vary orientation.
        """
        h = cavity + 2
        w = cavity + 3  # mouth two cells wide

        pat = np.full((h, w), "wall", dtype=object)
        pat[1:-1, 1:-1] = "empty"  # hollow centre
        pat[1:-1, -2] = "wall"     # close the back
        pat[1:-1, 0] = "empty"     # leave 2-cell mouth open

        return np.rot90(pat, self._rng.integers(4))

    # ------------------------------------------------------------------ #
    # Venus-fly-traps                                                   #
    # ------------------------------------------------------------------ #
    def _scatter_traps(self, grid: np.ndarray) -> None:
        n_traps = self._rng.integers(*self._trap_cnt)
        for _ in range(n_traps):
            if self._occ.mean() >= self._occ_thr:
                break
            depth = int(self._rng.integers(*self._trap_depth))
            width = int(self._rng.integers(*self._trap_width)) | 1  # force odd
            pattern = self._make_flytrap(depth, width)
            self._try_place(grid, pattern, clearance=1)

    def _make_flytrap(self, depth: int, width: int) -> np.ndarray:
        """
        Two blocks form a funnel with a 1-cell throat at the entrance.
        """
        h, w = depth + 2, width
        pat = np.full((h, w), "empty", dtype=object)

        # outer side walls
        pat[:, 0] = "wall"
        pat[:, -1] = "wall"

        # throat row
        pat[0, 2:-2] = "wall"

        # interior staggered walls
        for r in range(1, depth):
            if r % 2:
                pat[r, 2:-2] = "wall"
            else:
                pat[r, 3:-3] = "wall"

        pat[-1, :] = "wall"  # bottom seal
        return np.rot90(pat, self._rng.integers(4))

    # ------------------------------------------------------------------ #
    # Placement utilities                                                #
    # ------------------------------------------------------------------ #
    def _try_place(self, grid: np.ndarray, pattern: np.ndarray, *, clearance: int) -> bool:
        ph, pw = pattern.shape
        spots = self._candidates((ph, pw), clearance)
        if not spots:
            return False
        r, c = spots[self._rng.integers(len(spots))]
        grid[r : r + ph, c : c + pw] = pattern
        self._occ[r - clearance : r + ph + clearance, c - clearance : c + pw + clearance] = True
        return True

    def _candidates(self, shape: Tuple[int, int], clearance: int) -> List[Tuple[int, int]]:
        ph, pw = shape
        H, W = self._occ.shape
        if ph + 2 * clearance > H or pw + 2 * clearance > W:
            return []
        ok: list[Tuple[int, int]] = []
        for r in range(clearance, H - ph - clearance + 1):
            for c in range(clearance, W - pw - clearance + 1):
                if not self._occ[
                    r - clearance : r + ph + clearance,
                    c - clearance : c + pw + clearance
                ].any():
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
            for _ in range(int(cnt)):
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
