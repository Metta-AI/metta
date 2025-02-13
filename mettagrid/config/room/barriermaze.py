from typing import List, Dict
import random
import numpy as np
from omegaconf import DictConfig
from mettagrid.config.room.room import Room

class BarrierMaze(Room):
    """
    An environment with an outer wall border and three barriers placed among floor objects.

    The floor area (inside the border) is partitioned into seven blocks:
      1. Agent start ("agent.agent")
      2. Barrier 1
      3. Generator ("generator")
      4. Barrier 2
      5. Converter ("converter")
      6. Barrier 3
      7. Altar ("altar")

    Extra space between blocks is evenly distributed.

    New parameters:
      - barrier_placement_mode: "same", "alternating", or "doorways"
      - barrier_orientation: "vertical" or "horizontal"

    In this implementation the maze is built in a canonical vertical orientation.
    If barrier_orientation is "horizontal", we build using swapped dimensions and then
    rotate the final grid so that the floor items and barriers are positioned correctly.
    """
    def __init__(
        self,
        width: int,
        height: int,
        barrier_heights: List[int] = None,  # interpreted as sizes (heights or lengths)
        barrier_width: int = 1,
        num_barriers: int = 3,
        agents: int | DictConfig = 1,
        border_width: int = 1,
        border_object: str = "wall",
        barrier_placement_mode: str = "same",    # "same", "alternating", "doorways"
        barrier_orientation: str = "vertical"      # "vertical" or "horizontal"
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

        # For vertical, barrier "size" is measured along the vertical (secondary) axis.
        # For horizontal, barrier sizes are meant to be measured along the horizontal.
        # When building horizontally, we'll swap width and height so that the
        # canonical builder always treats the second dimension as the barrier axis.
        if self._orientation == "vertical":
            max_size = height - (2 * border_width) - 2
        else:
            max_size = width - (2 * border_width) - 2

        if barrier_heights is None:
            self._barrier_sizes = [random.randint(3, max_size + 1) for _ in range(num_barriers)]
        else:
            assert len(barrier_heights) == num_barriers, "Barrier sizes must match number of barriers."
            self._barrier_sizes = [min(s, max_size) for s in barrier_heights]

    def _build(self) -> np.ndarray:
        # Build the maze in a canonical (vertical) orientation.
        # For vertical mode, use the given width/height.
        # For horizontal mode, swap width and height then rotate the result.
        if self._orientation == "vertical":
            grid = self._build_canonical(self._width, self._height)
        else:
            # Build with swapped dimensions: primary axis (x) will be built from the original height.
            grid_swapped = self._build_canonical(self._height, self._width)
            # Rotate 90° clockwise so that the "floor" (built at the bottom) ends up on the left.
            grid = np.rot90(grid_swapped, k=-1)
        return grid

    def _build_canonical(self, grid_width: int, grid_height: int) -> np.ndarray:
        """
        Build the maze in a canonical (vertical) orientation:
          - Blocks (agent, barriers, generator, etc.) are arranged along the x-axis.
          - Floor items are placed on the bottom row.
          - Barriers extend vertically from either the top, bottom, or both (doorways).
        """
        # Initialize grid and define interior bounds.
        grid = np.full((grid_height, grid_width), "empty", dtype='<U50')
        bw = self._border_width
        interior_x_start, interior_x_end = bw, grid_width - bw
        interior_y_start, interior_y_end = bw, grid_height - bw

        # Draw outer border.
        grid[:interior_y_start, :] = self._border_object   # top border
        grid[interior_y_end:, :] = self._border_object      # bottom border
        grid[:, :interior_x_start] = self._border_object    # left border
        grid[:, interior_x_end:] = self._border_object      # right border

        # Helper to compute starting positions for blocks along the x-axis.
        def compute_positions(start: int, end: int, blocks: List[tuple]) -> Dict[str, int]:
            total_blocks = sum(width for _, width in blocks)
            total_gap = (end - start) - total_blocks
            num_gaps = len(blocks) - 1
            base_gap, extra = (total_gap // num_gaps, total_gap % num_gaps) if num_gaps > 0 else (0, 0)
            positions = {}
            pos = start
            for i, (name, width) in enumerate(blocks):
                positions[name] = pos
                pos += width
                if i < len(blocks) - 1:
                    pos += base_gap + (1 if i < extra else 0)
            return positions

        # Define blocks along the x-axis.
        blocks = [
            ("agent", 1),
            ("barrier1", self._barrier_width),
            ("generator", 1),
            ("barrier2", self._barrier_width),
            ("converter", 1),
            ("barrier3", self._barrier_width),
            ("altar", 1),
        ]
        positions = compute_positions(interior_x_start, interior_x_end, blocks)

        # Place floor items on the bottom row.
        floor_y = interior_y_end - 1
        grid[floor_y, positions["agent"]] = "agent.agent"
        grid[floor_y, positions["generator"]] = "generator"
        grid[floor_y, positions["converter"]] = "converter"
        grid[floor_y, positions["altar"]] = "altar"

        # Determine the available vertical (secondary) length.
        secondary_length = interior_y_end - interior_y_start

        # Draw each barrier.
        for idx, barrier_name in enumerate(["barrier1", "barrier2", "barrier3"]):
            size = self._barrier_sizes[idx]
            x = positions[barrier_name]
            if self._placement_mode == "alternating":
                if idx % 2 == 0:  # Even-indexed: attach at top.
                    top = interior_y_start
                    bottom = top + size - 1
                else:  # Odd-indexed: attach at bottom.
                    bottom = interior_y_end - 1
                    top = bottom - size + 1
                grid[top:bottom+1, x:x+self._barrier_width] = self._border_object

            elif self._placement_mode == "doorways":
                # Ensure a passage remains in the middle by capping barrier length.
                capped = min(size, max(1, (secondary_length - 1) // 2))
                # Top segment.
                top_seg_end = interior_y_start + capped - 1
                grid[interior_y_start:top_seg_end+1, x:x+self._barrier_width] = self._border_object
                # Bottom segment.
                bottom_seg_start = interior_y_end - capped
                grid[bottom_seg_start:interior_y_end, x:x+self._barrier_width] = self._border_object

            else:  # "same" mode: attach at the bottom.
                bottom = interior_y_end - 1
                top = bottom - size + 1
                grid[top:bottom+1, x:x+self._barrier_width] = self._border_object

        return grid
