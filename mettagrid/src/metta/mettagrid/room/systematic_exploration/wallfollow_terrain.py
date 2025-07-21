"""
Wall‑Follow Trap Terrain
========================
Generates a single‑room grid deliberately *unfriendly* to
memory‑less wall‑following (left‑hand / right‑hand) agents.

Design principles
-----------------
* **Disconnected islands.**  No obstacle touches the outer boundary,
  so a right‑/left‑hand rule keeps the agent circling the first island it meets.
* **Jagged perimeters.**  Every island is grown with random protrusions—there
  are *always* blocks sticking out, so the walls are never smooth.
* **Parameterisation.**  Island count, island size, occupancy
  threshold and arena size are exposed as constructor arguments so you
  can sweep them from YAML without touching Python code.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room


class WallFollowTerrain(Room):
    """
    A Room whose obstacle layout is adversarial for wall‑following.
    """

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        width: int = 60,
        height: int = 60,
        objects: DictConfig | dict | None = None,
        agents: int = 1,
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: str = "wall",
        #
        occupancy_threshold: float = 0.55,
        island_count: int = 45,
        island_size: int = 18,
    ) -> None:
        super().__init__(
            border_width=border_width,
            border_object=border_object,
            labels=["wallfollow"],
        )
        self.set_size_labels(width, height)

        self._H: int = height
        self._W: int = width
        self._rng = np.random.default_rng(seed)
        self._agents = agents if isinstance(agents, int) else 1
        self._objects = objects or {}
        self._occupancy_threshold = occupancy_threshold
        self._island_count = island_count
        self._island_size = island_size

        # occupancy mask (True ⇔ blocked)
        self._occ = np.zeros((height, width), dtype=bool)

    # ------------------------------------------------------------------ #
    # Room API                                                            #
    # ------------------------------------------------------------------ #
    def _build(self) -> np.ndarray:  # noqa: D401
        """
        Construct and return the textual grid.
        """
        grid = np.full((self._H, self._W), "empty", dtype=object)

        # 1. scatter disconnected, jagged islands
        self._scatter_islands(grid)

        # 2. spawn agents
        for _ in range(self._agents):
            pos = self._pick_empty()
            if pos is None:
                break
            r, c = pos
            grid[r, c] = "agent.agent"
            self._occ[r, c] = True

        # 3. user‑supplied objects (e.g. altars, hearts, etc.)
        for name, cnt in self._objects.items():
            for _ in range(cnt):
                pos = self._pick_empty()
                if pos is None:
                    break
                r, c = pos
                grid[r, c] = name
                self._occ[r, c] = True

        size = self._H * self._W
        if size > 7000:
            label = "large"
        elif size > 4500:
            label = "medium"
        else:
            label = "small"

        np.save(f"terrains/wallfollow_terrain/{label}_{self._rng.integers(1000000)}.npy", grid)

        return grid

    # ------------------------------------------------------------------ #
    # Island generation                                                   #
    # ------------------------------------------------------------------ #
    def _scatter_islands(self, grid: np.ndarray) -> None:
        for _ in range(self._island_count):
            if self._occ.mean() >= self._occupancy_threshold:
                break
            pattern = self._make_jagged_blob(self._island_size)
            self._try_place(grid, pattern, clearance=1)

    def _make_jagged_blob(self, target_cells: int) -> np.ndarray:
        """
        Grow a connected blob of `target_cells` and add protrusions so
        no edge remains smooth for more than 2 cells.
        """
        cells = {(0, 0)}
        while len(cells) < target_cells:
            frontier = [
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if (r + dr, c + dc) not in cells
            ]
            if not frontier:
                break
            cells.add(frontier[self._rng.integers(len(frontier))])

        # add outward‑facing spikes on ≈25 % of perimeter neighbours
        perimeter = [
            (r + dr, c + dc)
            for r, c in cells
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if (r + dr, c + dc) not in cells
        ]
        if perimeter:
            spikes = self._rng.choice(
                perimeter,
                size=max(1, len(perimeter) // 4),
                replace=False,
            )
            cells.update(tuple(spike) for spike in spikes)

        # convert set → tight ndarray pattern
        rs, cs = zip(*cells, strict=False)
        pattern = np.full(
            (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1),
            "empty",
            dtype=object,
        )
        for r, c in cells:
            pattern[r - min(rs), c - min(cs)] = "wall"
        return pattern

    # ------------------------------------------------------------------ #
    # Spatial utilities                                                   #
    # ------------------------------------------------------------------ #
    def _pick_empty(self) -> Optional[Tuple[int, int]]:
        flat = np.flatnonzero(~self._occ)
        if flat.size == 0:
            return None
        idx = self._rng.integers(flat.size)
        return np.unravel_index(flat[idx], self._occ.shape)

    def _try_place(self, grid: np.ndarray, pattern: np.ndarray, *, clearance: int = 0) -> bool:
        ph, pw = pattern.shape
        candidates = self._find_candidates((ph, pw), clearance)
        if not candidates:
            return False
        r, c = candidates[self._rng.integers(len(candidates))]
        grid[r : r + ph, c : c + pw] = pattern
        self._occ[r : r + ph, c : c + pw] |= pattern == "wall"
        return True

    # sliding‑window emptiness test
    def _find_candidates(self, shape: Tuple[int, int], clearance: int) -> List[Tuple[int, int]]:
        ph, pw = shape
        H, W = self._occ.shape
        if ph + 2 * clearance > H or pw + 2 * clearance > W:
            return []
        occ_int = self._occ.astype(int).cumsum(0).cumsum(1)

        def area_sum(r0: int, c0: int, r1: int, c1: int) -> int:
            total = occ_int[r1, c1]
            if r0 > 0:
                total -= occ_int[r0 - 1, c1]
            if c0 > 0:
                total -= occ_int[r1, c0 - 1]
            if r0 > 0 and c0 > 0:
                total += occ_int[r0 - 1, c0 - 1]
            return total

        cand: list[Tuple[int, int]] = []
        for r in range(clearance, H - ph - clearance + 1):
            for c in range(clearance, W - pw - clearance + 1):
                if (
                    area_sum(
                        r - clearance,
                        c - clearance,
                        r + ph + clearance - 1,
                        c + pw + clearance - 1,
                    )
                    == 0
                ):
                    cand.append((r, c))
        return cand


# if __name__ == "__main__":
#     for i in range(500):
#         width = np.random.randint(40, 120)
#         height = np.random.randint(40, 120)
#         room = WallFollowTerrain(width=width, height=height)
#         room.build()
