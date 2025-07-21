"""
Wall‑Follow *Sparse* Terrain
============================
A huge, otherwise empty room whose perimeter is **crenellated**:
`wall, gap, wall, gap, …` along every edge.

* A right‑ or left‑hand wall‑follower can never keep continuous
  contact—it repeatedly loses the wall at each gap.
* Interior is completely open except for optional *altar* objects.

Exposed YAML parameters
-----------------------
width, height, agents, seed (usual)
objects: {altar: <count>}          # number of altars to scatter
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room


class WallFollowSparseTerrain(Room):
    """Large empty arena with crenellated border to defeat wall‑following."""

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        width: int = 120,
        height: int = 120,
        agents: int = 1,
        objects: DictConfig | Dict[str, int] | None = None,
        seed: Optional[int] = None,
        border_object: str = "wall",
    ) -> None:
        # We set border_width = 0 and create the border pattern ourselves.
        super().__init__(border_width=0, border_object=border_object, labels=["wallfollow_sparse"])
        self.set_size_labels(width, height)

        self._H, self._W = height, width
        self._rng = np.random.default_rng(seed)
        self._agents = agents if isinstance(agents, int) else 1
        self._objects = {} if objects is None else dict(objects)

        # occupancy mask: True ⇔ blocked or filled
        self._occ = np.zeros((height, width), dtype=bool)

    # ------------------------------------------------------------------ #
    # Room API                                                            #
    # ------------------------------------------------------------------ #
    def _build(self) -> np.ndarray:
        grid = np.full((self._H, self._W), "empty", dtype=object)

        # 1 ─ crenellated perimeter
        self._build_crenellated_border(grid)

        # 2 ─ scatter agents
        for _ in range(self._agents):
            pos = self._pick_empty_interior()
            if pos is None:
                break
            r, c = pos
            grid[r, c] = "agent.agent"
            self._occ[r, c] = True

        # 3 ─ scatter user‑requested objects (e.g. altars)
        for name, cnt in self._objects.items():
            for _ in range(int(cnt)):
                pos = self._pick_empty_interior()
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

        np.save(f"terrains/wallfollow_sparse/{label}_{self._rng.integers(1000000)}.npy", grid)

        return grid

    # ------------------------------------------------------------------ #
    # Border construction                                                 #
    # ------------------------------------------------------------------ #
    def _build_crenellated_border(self, grid: np.ndarray) -> None:
        # Top & bottom rows
        for c in range(self._W):
            if c % 2 == 0:  # start with wall, alternate
                grid[0, c] = grid[-1, c] = "wall"
                self._occ[0, c] = self._occ[-1, c] = True

        # Left & right columns
        for r in range(self._H):
            if r % 2 == 0:
                grid[r, 0] = grid[r, -1] = "wall"
                self._occ[r, 0] = self._occ[r, -1] = True

    # ------------------------------------------------------------------ #
    # Utilities                                                           #
    # ------------------------------------------------------------------ #
    def _pick_empty_interior(self) -> Optional[Tuple[int, int]]:
        """
        Return a random empty cell strictly *inside* the perimeter.
        """
        interior_mask = ~self._occ.copy()
        interior_mask[0, :] = interior_mask[-1, :] = False
        interior_mask[:, 0] = interior_mask[:, -1] = False

        empty_flat = np.flatnonzero(interior_mask)
        if empty_flat.size == 0:
            return None
        idx = self._rng.integers(empty_flat.size)
        return np.unravel_index(empty_flat[idx], self._occ.shape)


# if __name__ == "__main__":
#     for i in range(500):
#         width = np.random.randint(40, 120)
#         height = np.random.randint(40, 120)
#         room = WallFollowSparseTerrain(width=width, height=height)
#         room.build()
