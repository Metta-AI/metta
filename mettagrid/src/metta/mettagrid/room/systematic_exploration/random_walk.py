"""
Random‑Walk Trap Terrain
========================
Builds an arena that is *slow to cover* for a pure random‑walk agent:

* A *huge open plaza* (almost no obstacles) – cover‑time Θ(A log A).
* Dozens of *deep, one‑way corridors* (dead‑ends) branching from the plaza.
  The expected escape time from a length‑L cul‑de‑sac grows ≈ L².

All parameters are configurable from the calling YAML.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room


class RandomWalkTerrain(Room):
    """Room builder that frustrates memory‑less random walkers."""

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
        occupancy_threshold: float = 0.30,  # keep the map mostly open
        corridor_count: int = 35,  # how many cul‑de‑sacs
        corridor_length: int = 50,  # inclusive range
    ) -> None:
        super().__init__(border_width=border_width, border_object=border_object, labels=["random_walk"])
        self.set_size_labels(width, height)

        self._H, self._W = height, width
        self._rng = np.random.default_rng(seed)
        self._agents = agents if isinstance(agents, int) else 1
        self._objects = objects or {}

        self._occ = np.zeros((height, width), dtype=bool)
        self._occ_thr = occupancy_threshold
        self._corridor_count = corridor_count
        self._corridor_len = corridor_length

    # ------------------------------------------------------------------ #
    # Public Room API                                                    #
    # ------------------------------------------------------------------ #
    def _build(self) -> np.ndarray:
        grid = np.full((self._H, self._W), "empty", dtype=object)

        # 1. carve dead‑end corridors
        self._dig_corridors(grid)

        # 2. spawn agents
        for _ in range(self._agents):
            pos = self._pick_empty()
            if pos is None:
                break
            r, c = pos
            grid[r, c] = "agent.agent"
            self._occ[r, c] = True

        # 3. optional goal / resource objects
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

        np.save(f"terrains/random_walk/{label}_{self._rng.integers(1000000)}.npy", grid)

        return grid

    # ------------------------------------------------------------------ #
    # Corridor carving                                                   #
    # ------------------------------------------------------------------ #
    def _dig_corridors(self, grid: np.ndarray) -> None:
        N = self._corridor_count
        for _ in range(N):
            if self._occ.mean() >= self._occ_thr:
                break
            length = self._corridor_len
            orient = self._rng.choice(["h", "v"])

            if orient == "h":
                r = self._rng.integers(2, self._H - 2)
                c0 = self._rng.integers(2, self._W - length - 2)
                self._carve_horizontal(grid, r, c0, length)
            else:
                c = self._rng.integers(2, self._W - 2)
                r0 = self._rng.integers(2, self._H - length - 2)
                self._carve_vertical(grid, r0, c, length)

    def _carve_horizontal(self, grid: np.ndarray, r: int, c0: int, L: int) -> None:
        """Create a horizontal cul‑de‑sac opening to the left."""
        walkway_cols = range(c0, c0 + L)  # L cells of empty walkway
        # Walls above & below
        grid[r - 1, walkway_cols] = "wall"
        grid[r + 1, walkway_cols] = "wall"
        self._occ[r - 1, walkway_cols] = True
        self._occ[r + 1, walkway_cols] = True
        # Walkway empty
        self._occ[r, walkway_cols] = False
        # Close the far end
        grid[r, c0 + L - 1] = "wall"
        self._occ[r, c0 + L - 1] = True

    def _carve_vertical(self, grid: np.ndarray, r0: int, c: int, L: int) -> None:
        """Create a vertical cul‑de‑sac opening upwards."""
        walkway_rows = range(r0, r0 + L)
        grid[walkway_rows, c - 1] = "wall"
        grid[walkway_rows, c + 1] = "wall"
        self._occ[walkway_rows, c - 1] = True
        self._occ[walkway_rows, c + 1] = True
        self._occ[walkway_rows, c] = False
        grid[r0 + L - 1, c] = "wall"
        self._occ[r0 + L - 1, c] = True

    # ------------------------------------------------------------------ #
    # Utilities                                                          #
    # ------------------------------------------------------------------ #
    def _pick_empty(self) -> Optional[Tuple[int, int]]:
        empty_flat = np.flatnonzero(~self._occ)
        if empty_flat.size == 0:
            return None
        idx = self._rng.integers(empty_flat.size)
        return np.unravel_index(empty_flat[idx], self._occ.shape)


# if __name__ == "__main__":
#     for i in range(500):
#         width = np.random.randint(50, 120)
#         height = np.random.randint(50, 120)
#         occupancy_threshold = np.random.randint(1, 3) / 10
#         corridor_count = np.random.randint(15, 35)
#         corridor_length = np.random.randint(12, 30)
#         room = RandomWalkTerrain(
#             width=width,
#             height=height,
#             occupancy_threshold=occupancy_threshold,
#             corridor_count=corridor_count,
#             corridor_length=corridor_length,
#         )
#         room.build()
