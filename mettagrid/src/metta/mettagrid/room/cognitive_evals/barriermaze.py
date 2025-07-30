import random
from typing import List

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room
from metta.mettagrid.room.utils import compute_positions, create_grid, draw_border


class BarrierMaze(Room):
    """
    Maze with an outer wall and three barriers separating seven blocks:
      1. Agent start ("agent.agent")
      2. Barrier 1
      3. mine ("mine")
      4. Barrier 2
      5. generator ("generator")
      6. Barrier 3
      7. Altar ("altar")

    Parameters:
      - barrier_placement_mode: "same", "alternating", or "doorways"
      - barrier_orientation: "vertical" or "horizontal"

    For horizontal orientation, dimensions are swapped and the final grid is rotated.
    """

    def __init__(
        self,
        width: int,
        height: int,
        barrier_heights: List[int] = None,  # sizes along the secondary axis
        barrier_width: int = 1,
        num_barriers: int = 3,
        agents: int | DictConfig = 1,
        border_width: int = 1,
        border_object: str = "wall",
        barrier_placement_mode: str = "same",  # "same", "alternating", "doorways"
        barrier_orientation: str = "vertical",  # "vertical" or "horizontal"
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._width = width
        self._height = height
        self._barrier_width = barrier_width
        self._agents = agents
        self._border_width = border_width
        self._border_object = border_object
        self._placement_mode = barrier_placement_mode.lower()
        self._orientation = barrier_orientation.lower()

        # Maximum barrier size is along the secondary axis.
        max_size = (height if self._orientation == "vertical" else width) - 2 * border_width - 2
        if barrier_heights is None:
            self._barrier_sizes = [random.randint(3, max_size + 1) for _ in range(num_barriers)]
        else:
            assert len(barrier_heights) == num_barriers, "Barrier sizes must match number of barriers."
            self._barrier_sizes = [min(s, max_size) for s in barrier_heights]

    def _build(self) -> np.ndarray:
        if self._orientation == "vertical":
            grid = self._build_canonical(self._width, self._height)
        else:
            # Build with swapped dimensions and rotate 90° clockwise.
            grid = np.rot90(self._build_canonical(self._height, self._width), k=-1)
        return grid

    def _build_canonical(self, grid_width: int, grid_height: int) -> np.ndarray:
        # Create grid and draw border using utility functions.
        grid = create_grid(grid_height, grid_width, fill_value="empty")
        draw_border(grid, self._border_width, self._border_object)

        interior_x_start, interior_x_end = self._border_width, grid_width - self._border_width
        interior_y_start, interior_y_end = self._border_width, grid_height - self._border_width

        # Define blocks along the x-axis.
        blocks = [
            ("agent", 1),
            ("barrier1", self._barrier_width),
            ("mine", 1),
            ("barrier2", self._barrier_width),
            ("generator", 1),
            ("barrier3", self._barrier_width),
            ("altar", 1),
        ]
        positions = compute_positions(interior_x_start, interior_x_end, blocks)

        # Place floor items on the bottom row.
        floor_y = interior_y_end - 1
        grid[floor_y, positions["agent"]] = "agent.agent"
        grid[floor_y, positions["mine"]] = "mine"
        grid[floor_y, positions["generator"]] = "generator"
        grid[floor_y, positions["altar"]] = "altar"

        secondary_length = interior_y_end - interior_y_start

        # Draw barriers.
        for idx, barrier in enumerate(["barrier1", "barrier2", "barrier3"]):
            size = self._barrier_sizes[idx]
            x = positions[barrier]
            if self._placement_mode in ("alternating", "same"):
                # For alternating mode, even-indexed barriers attach at the top; otherwise, at the bottom.
                top = (
                    interior_y_start
                    if (self._placement_mode == "alternating" and idx % 2 == 0)
                    else interior_y_end - size
                )
                grid[top : top + size, x : x + self._barrier_width] = self._border_object
            elif self._placement_mode == "doorways":
                capped = min(size, max(1, (secondary_length - 1) // 2))
                grid[interior_y_start : interior_y_start + capped, x : x + self._barrier_width] = self._border_object
                grid[interior_y_end - capped : interior_y_end, x : x + self._barrier_width] = self._border_object
        return grid
