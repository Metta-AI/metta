from typing import List, Optional, Tuple

import numpy as np

from metta.mettagrid.room.room import Room


class CylinderWorld(Room):
    def __init__(
        self,
        width: int,
        height: int,
        agents: int | dict = 0,
        seed: Optional[int] = 42,
        border_width: int = 0,
        border_object: str = "wall",
        team: str | None = None,
    ):
        super().__init__(border_width=border_width, border_object=border_object, labels=["cylinder_world"])
        self._rng = np.random.default_rng(seed)
        width, height = np.random.randint(40, 100), np.random.randint(40, 100)
        self._width, self._height = width, height
        self._agents = agents
        self._team = team
        # occupancy mask: False = empty
        self._occ = np.zeros((height, width), dtype=bool)

        self.set_size_labels(width, height)

    # ------------------------------------------------------------------ #
    # Public build
    # ------------------------------------------------------------------ #
    def _build(self) -> np.ndarray:
        return self._build_cylinder_world()

    # ------------------------------------------------------------------ #
    # Cylinder‑only build
    # ------------------------------------------------------------------ #
    def _build_cylinder_world(self) -> np.ndarray:
        """
        Keep adding cylinders until *no* size/orientation fits anywhere.
        Strategy: restart the attempt with a fresh random cylinder after every
        successful placement. Stop only after ``max_consecutive_fail`` failed
        attempts *in a row* (i.e. we tried many random sizes/orientations
        without success), which strongly suggests the map is packed.
        """
        grid = np.full((self._height, self._width), "empty", dtype=object)
        self._occ[:, :] = False

        max_consecutive_fail = 2
        fails = 0
        while fails < max_consecutive_fail:
            placed = self._place_cylinder_once(grid, clearance=1)
            if placed:
                fails = 0  # reset – we still found room
            else:
                fails += 1  # try a different size/orientation

        # Finally, spawn any requested agents on leftover empty cells
        grid = self._place_agents(grid)
        return grid

    # ------------------------------------------------------------------ #
    # Cylinder placement helpers
    # ------------------------------------------------------------------ #
    def _place_cylinder_once(self, grid: np.ndarray, clearance: int = 1) -> bool:
        pat = self._generate_cylinder_pattern()
        return self._place_region(grid, pat, clearance)

    def _generate_cylinder_pattern(self) -> np.ndarray:
        length = int(self._rng.integers(2, 30))
        gap = int(self._rng.integers(1, 4))
        vertical = bool(self._rng.integers(0, 2))
        if vertical:
            h, w = length, gap + 2
            pat = np.full((h, w), "empty", dtype=object)
            pat[:, 0] = pat[:, -1] = "wall"
            pat[h // 2, 1 + gap // 2] = "altar"
        else:
            h, w = gap + 2, length
            pat = np.full((h, w), "empty", dtype=object)
            pat[0, :] = pat[-1, :] = "wall"
            pat[1 + gap // 2, w // 2] = "altar"
        return pat

    # ------------------------------------------------------------------ #
    # Agents placement (simplified)
    # ------------------------------------------------------------------ #
    def _place_agents(self, grid):
        if self._team is None:
            agents = ["agent.agent"] * self._agents
        else:
            agents = ["agent." + self._team] * self._agents
        for a in agents:
            pos = self._rand_empty()
            if pos:
                grid[pos] = a
                self._occ[pos] = True
        return grid

    # ------------------------------------------------------------------ #
    # Region placement utilities
    # ------------------------------------------------------------------ #
    def _place_region(self, grid, pattern: np.ndarray, clearance: int) -> bool:
        ph, pw = pattern.shape
        for r, c in self._candidate_positions((ph + 2 * clearance, pw + 2 * clearance)):
            grid[r + clearance : r + clearance + ph, c + clearance : c + clearance + pw] = pattern
            self._occ[r + clearance : r + clearance + ph, c + clearance : c + clearance + pw] |= pattern != "empty"
            return True
        return False

    def _candidate_positions(self, shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        h, w = shape
        H, W = self._occ.shape
        if H < h or W < w:
            return []
        view_shape = (H - h + 1, W - w + 1, h, w)
        sub = np.lib.stride_tricks.as_strided(self._occ, view_shape, self._occ.strides * 2)
        sums = sub.sum(axis=(2, 3))
        coords = np.argwhere(sums == 0)
        self._rng.shuffle(coords)
        return [tuple(x) for x in coords]

    def _rand_empty(self) -> Optional[Tuple[int, int]]:
        empties = np.flatnonzero(~self._occ)
        if not len(empties):
            return None
        idx = self._rng.integers(0, len(empties))
        return tuple(np.unravel_index(empties[idx], self._occ.shape))
