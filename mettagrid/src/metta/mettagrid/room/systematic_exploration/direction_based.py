"""
Direction‑Biased Random‑Walk Trap Terrain
========================================
A single‑room layout that causes forward‑favouring memory‑less agents to
oscillate endlessly:

* **Labyrinth of parallel corridors** – 1‑cell corridors separated by 1‑cell
  walls; very few cross‑connections, so an agent that keeps the same heading
  ping‑pongs forever inside one corridor.
* **U‑shaped bays with long prongs** – deep cul‑de‑sacs attached side‑on to
  corridors; once inside, an agent repeatedly bounces between prongs without
  escaping.

All geometry parameters are configurable from YAML.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room


class DirectionBasedTerrain(Room):
    """
    Terrain generator adversarial to forward‑biased random walks.
    """

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
        hole_count: int = 3,  # holes per wall column
        bay_count: int = 20,  # how many U‑bays
        bay_depth: int = 8,  # vertical depth (cells)
        bay_width: int = 4,  # horizontal span
    ) -> None:
        super().__init__(
            border_width=border_width,
            border_object=border_object,
            labels=["direction_based"],
        )
        self.set_size_labels(width, height)

        self._H, self._W = height, width
        self._rng = np.random.default_rng(seed)
        self._agents = agents if isinstance(agents, int) else 1
        self._objects = objects or {}

        # Parameters
        self._hole_count = hole_count
        self._bay_count = bay_count
        self._bay_depth = bay_depth
        self._bay_width = bay_width

    # ------------------------------------------------------------------ #
    # Room API                                                            #
    # ------------------------------------------------------------------ #
    def _build(self) -> np.ndarray:
        # Start with an empty grid.
        grid = np.full((self._H, self._W), "empty", dtype=object)

        # Step 1 ─ Parallel‑corridor labyrinth.
        self._make_vertical_labyrinth(grid)

        # Step 2 ─ Attach U‑shaped bays.
        self._attach_u_bays(grid)

        # Build occupancy mask *after* terrain carving.
        occ = grid != "empty"

        # Step 3 ─ Spawn agents.
        for _ in range(self._agents):
            pos = self._pick_empty(occ)
            if pos is None:
                break
            r, c = pos
            grid[r, c] = "agent.agent"
            occ[r, c] = True

        # Step 4 ─ Optional task objects (altars, etc.).
        for name, cnt in self._objects.items():
            for _ in range(cnt):
                pos = self._pick_empty(occ)
                if pos is None:
                    break
                r, c = pos
                grid[r, c] = name
                occ[r, c] = True

        # self._scatter_agents_and_objects(grid)
        size = self._H * self._W
        if size > 7000:
            label = "large"
        elif size > 4500:
            label = "medium"
        else:
            label = "small"

        np.save(f"terrains/direction_based/{label}_{self._rng.integers(1000000)}.npy", grid)

        return grid

    # ------------------------------------------------------------------ #
    # Terrain primitives                                                  #
    # ------------------------------------------------------------------ #
    def _make_vertical_labyrinth(self, grid: np.ndarray) -> None:
        """
        Fill every odd column with walls, then punch small random holes so the
        whole arena is technically connected but cross‑corridor transfers are
        *rare*.
        """
        # Odd columns (1,3,5,…) become thin walls.
        grid[:, 1::2] = "wall"

        # For each wall column punch 3–8 holes.
        for col in range(1, self._W, 2):
            n_holes = self._hole_count
            for _ in range(n_holes):
                row = self._rng.integers(1, self._H - 1)
                grid[row, col] = "empty"

    def _attach_u_bays(self, grid: np.ndarray) -> None:
        """
        Randomly choose corridor cells and carve a U‑shaped cul‑de‑sac that
        sticks out to the right (east).  Each bay is:

            entry
              │
              │
            ┌─┴─────┐
            │       │   depth × width (∽ rectangle)
            └───────┘

        """
        n_bays = self._bay_count

        for _ in range(n_bays):
            depth = self._bay_depth
            width = self._bay_width
            # Pick a corridor column (even), leaving margin so the bay fits.
            entry_col_candidates = [c for c in range(0, self._W - width - 2, 2) if c + width + 1 < self._W - 1]
            if not entry_col_candidates:
                break
            entry_col = entry_col_candidates[self._rng.integers(len(entry_col_candidates))]

            # Pick an entry row so the bay fits vertically.
            entry_row = self._rng.integers(1, self._H - depth - 2)

            self._carve_u(grid, entry_row, entry_col, depth, width)

    # ------------------------------------------------------------------ #
    # Geometry helpers                                                   #
    # ------------------------------------------------------------------ #
    def _carve_u(
        self,
        grid: np.ndarray,
        r: int,
        c: int,
        depth: int,
        width: int,
    ) -> None:
        """
        Carve a U‑shaped bay whose entry point is (r, c).
        Orientation: faces east (to the right).
        """
        # Horizontal top segment (entry row).
        for dc in range(1, width + 1):
            grid[r, c + dc] = "empty"
        # Two vertical prongs.
        left_leg = c + 1
        right_leg = c + width
        for dr in range(1, depth + 1):
            grid[r + dr, left_leg] = "empty"
            grid[r + dr, right_leg] = "empty"
        # Bottom connector.
        bottom_row = r + depth
        for dc in range(1, width):
            grid[bottom_row, c + dc] = "empty"

    # ------------------------------------------------------------------ #
    # Utility                                                             #
    # ------------------------------------------------------------------ #
    def _pick_empty(self, occ: np.ndarray) -> Optional[Tuple[int, int]]:
        flat = np.flatnonzero(~occ)
        if flat.size == 0:
            return None
        idx = self._rng.integers(flat.size)
        return np.unravel_index(flat[idx], occ.shape)


# if __name__ == "__main__":
#     for i in range(500):
#         width = np.random.randint(40, 120)
#         height = np.random.randint(40, 120)
#         hole_count = np.random.randint(3, 8)
#         bay_count = np.random.randint(20, 40)
#         bay_depth = np.random.randint(5, 15)
#         bay_width = np.random.randint(5, 15)
#         room = DirectionBasedTerrain(
#             width=width,
#             height=height,
#             hole_count=hole_count,
#             bay_count=bay_count,
#             bay_depth=bay_depth,
#             bay_width=bay_width,
#         )
#         room.build()
