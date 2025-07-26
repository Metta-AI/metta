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
        hole_count: int = 3,
        bay_count: int = 20,
        bay_depth: int = 8,
        bay_width: int = 4,
        #
        orientation: str = "auto",  # "auto"| "horizontal" | "vertical"
    ) -> None:
        super().__init__(
            border_width=border_width,
            border_object=border_object,
            labels=["direction_based_sparse"],
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
        self._orientation_arg = orientation.lower()

    # ------------------------------------------------------------------ #
    # Room API                                                            #
    # ------------------------------------------------------------------ #
    def _build(self) -> np.ndarray:
        # Decide orientation *now*, not in __init__
        if self._orientation_arg == "horizontal":
            horizontal = True
        elif self._orientation_arg == "vertical":
            horizontal = False
        else:  # "auto"  (=50 : 50)
            horizontal = self._rng.random() < 0.5

        grid = np.full((self._H, self._W), "empty", dtype=object)

        # 1 ─ Corridor labyrinth
        if horizontal:
            self._make_horizontal_labyrinth(grid)
        else:
            self._make_vertical_labyrinth(grid)

        # 2 ─ U‑bays (matching orientation)
        if horizontal:
            self._attach_u_bays_horizontal(grid)
        else:
            self._attach_u_bays(grid)

        # 3 ─ Agents & objects
        occ = grid != "empty"
        self._scatter_agents_and_objects(grid, occ)

        # self._scatter_agents_and_objects(grid)
        size = self._H * self._W
        if size > 7000:
            label = "large"
        elif size > 4500:
            label = "medium"
        else:
            label = "small"

        np.save(f"terrains/direction_based_sparse/{label}_{self._rng.integers(1000000)}.npy", grid)

        return grid

    # ------------------------------------------------------------------ #
    # Agent / object placement                                           #
    # ------------------------------------------------------------------ #
    def _scatter_agents_and_objects(self, grid: np.ndarray, occ: np.ndarray) -> None:
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
        """Pick a random empty position from the grid."""
        flat = np.flatnonzero(~occ)
        if flat.size == 0:
            return None
        idx = self._rng.integers(flat.size)
        return np.unravel_index(flat[idx], occ.shape)

    # ------------------------------------------------------------------ #
    # Labyrinth generation                                               #
    # ------------------------------------------------------------------ #
    def _make_vertical_labyrinth(self, grid: np.ndarray) -> None:
        """Create a vertical corridor labyrinth with holes."""
        hole_count = self._hole_count

        # Create vertical corridors
        num_corridors = self._W // 8  # Adjust spacing based on width
        for i in range(num_corridors):
            col = (i + 1) * (self._W // (num_corridors + 1))
            if col < self._W - 1:
                # Create vertical corridor
                for row in range(1, self._H - 1):
                    if self._rng.random() < 0.8:  # 80% chance to place wall
                        grid[row, col] = "wall"

        # Add some horizontal connectors with holes
        for _ in range(hole_count):
            row = self._rng.integers(2, self._H - 2)
            start_col = self._rng.integers(1, self._W - 10)
            end_col = start_col + self._rng.integers(3, 8)
            end_col = min(end_col, self._W - 1)

            # Create horizontal corridor with gaps
            for col in range(start_col, end_col):
                if self._rng.random() < 0.3:  # 30% chance to leave gap
                    continue
                grid[row, col] = "wall"

    def _make_horizontal_labyrinth(self, grid: np.ndarray) -> None:
        """Create a horizontal corridor labyrinth with holes."""
        hole_count = self._hole_count

        # Create horizontal corridors
        num_corridors = self._H // 8  # Adjust spacing based on height
        for i in range(num_corridors):
            row = (i + 1) * (self._H // (num_corridors + 1))
            if row < self._H - 1:
                # Create horizontal corridor
                for col in range(1, self._W - 1):
                    if self._rng.random() < 0.8:  # 80% chance to place wall
                        grid[row, col] = "wall"

        # Add some vertical connectors with holes
        for _ in range(hole_count):
            col = self._rng.integers(2, self._W - 2)
            start_row = self._rng.integers(1, self._H - 10)
            end_row = start_row + self._rng.integers(3, 8)
            end_row = min(end_row, self._H - 1)

            # Create vertical corridor with gaps
            for row in range(start_row, end_row):
                if self._rng.random() < 0.3:  # 30% chance to leave gap
                    continue
                grid[row, col] = "wall"

    # ------------------------------------------------------------------ #
    # U-bay generation                                                   #
    # ------------------------------------------------------------------ #
    def _attach_u_bays(self, grid: np.ndarray) -> None:
        """Attach vertical U-shaped bays to the labyrinth."""
        bay_count = self._bay_count

        for _ in range(bay_count):
            # Pick a random position for the bay
            row = self._rng.integers(3, self._H - 3)
            col = self._rng.integers(3, self._W - 3)

            depth = self._bay_depth
            width = self._bay_width

            # Ensure bay fits within bounds
            depth = min(depth, self._H - row - 2)
            width = min(width, self._W - col - 2)

            # Create U-shaped bay (vertical orientation)
            # Left wall
            for r in range(row, row + depth):
                if r < self._H and col < self._W:
                    grid[r, col] = "wall"

            # Right wall
            for r in range(row, row + depth):
                if r < self._H and col + width < self._W:
                    grid[r, col + width] = "wall"

            # Bottom wall (closing the U)
            for c in range(col, col + width + 1):
                if row + depth < self._H and c < self._W:
                    grid[row + depth, c] = "wall"

    def _attach_u_bays_horizontal(self, grid: np.ndarray) -> None:
        """Attach horizontal U-shaped bays to the labyrinth."""
        bay_count = self._bay_count

        for _ in range(bay_count):
            # Pick a random position for the bay
            row = self._rng.integers(3, self._H - 3)
            col = self._rng.integers(3, self._W - 3)

            depth = self._bay_depth
            width = self._bay_width

            # Ensure bay fits within bounds
            depth = min(depth, self._W - col - 2)
            width = min(width, self._H - row - 2)

            # Create U-shaped bay (horizontal orientation)
            # Top wall
            for c in range(col, col + depth):
                if row < self._H and c < self._W:
                    grid[row, c] = "wall"

            # Bottom wall
            for c in range(col, col + depth):
                if row + width < self._H and c < self._W:
                    grid[row + width, c] = "wall"

            # Right wall (closing the U)
            for r in range(row, row + width + 1):
                if r < self._H and col + depth < self._W:
                    grid[r, col + depth] = "wall"


# if __name__ == "__main__":
#     for i in range(500):
#         width = np.random.randint(40, 150)
#         height = np.random.randint(40, 150)
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
