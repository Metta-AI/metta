from typing import List
import random
import numpy as np
from omegaconf import DictConfig
from mettagrid.config.room.room import Room

class BarrierMaze(Room):
    """
    An environment with a wall border and three vertical barriers along the interior's bottom row.

    The interior's bottom row is divided into seven blocks (from left to right):
      1. Agent start ("agent.agent")
      2. Barrier 1 (vertical wall)
      3. Generator ("generator")
      4. Barrier 2 (vertical wall)
      5. Converter ("converter")
      6. Barrier 3 (vertical wall)
      7. Altar ("altar")

    Extra horizontal space is evenly distributed as gaps between blocks.

    If 'alternating' is True, the barriers are placed alternatingly:
      - Even-indexed barriers attach to the top of the interior.
      - Odd-indexed barriers attach to the bottom of the interior.
    Otherwise, all barriers extend upward from the bottom row.
    """
    def __init__(
        self,
        width: int,
        height: int,
        barrier_heights: List[int] = None,
        barrier_width: int = 1,
        num_barriers: int = 3,
        agents: int | DictConfig = 1,
        border_width: int = 1,
        border_object: str = "wall",
        alternating: bool = False,
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._width = width
        self._height = height
        self._barrier_width = barrier_width
        self._agents = agents
        self._border_width = border_width
        self._border_object = border_object
        self._alternating = alternating

        max_barrier_height = height - (2 * border_width) - 2

        if barrier_heights is None:
            # Generate random barrier heights if not provided.
            self._barrier_heights = [
                random.randint(3, max_barrier_height + 1)
                for _ in range(num_barriers)
            ]
        else:
            assert len(barrier_heights) == num_barriers, "Barrier heights must match the number of barriers."
            self._barrier_heights = [min(h, max_barrier_height) for h in barrier_heights]

    def _build(self) -> np.ndarray:
        # Initialize grid with "empty" cells.
        grid = np.full((self._height, self._width), "empty", dtype='<U50')

        bw = self._border_width
        interior_x_start, interior_x_end = bw, self._width - bw
        interior_y_start, interior_y_end = bw, self._height - bw
        bottom_y = interior_y_end - 1
        interior_width = interior_x_end - interior_x_start

        # Define the seven blocks (name, width)
        blocks = [
            ("agent", 1),
            ("barrier1", self._barrier_width),
            ("generator", 1),
            ("barrier2", self._barrier_width),
            ("converter", 1),
            ("barrier3", self._barrier_width),
            ("altar", 1),
        ]
        total_block_width = sum(width for _, width in blocks)
        total_gap = interior_width - total_block_width
        assert total_gap >= 0, "Interior width is too small for the required arrangement."

        # Evenly distribute extra space between blocks.
        num_gaps = len(blocks) - 1
        base_gap, extra = divmod(total_gap, num_gaps) if num_gaps > 0 else (0, 0)

        x_positions = {}
        x = interior_x_start
        for i, (name, width) in enumerate(blocks):
            x_positions[name] = x
            x += width
            if i < len(blocks) - 1:
                x += base_gap + (1 if i < extra else 0)

        # Place floor items on the interior's bottom row.
        grid[bottom_y, x_positions["agent"]] = "agent.agent"
        grid[bottom_y, x_positions["generator"]] = "generator"
        grid[bottom_y, x_positions["converter"]] = "converter"
        grid[bottom_y, x_positions["altar"]] = "altar"

        # Draw each barrier.
        for idx, barrier_name in enumerate(["barrier1", "barrier2", "barrier3"]):
            h = self._barrier_heights[idx]
            x_start = x_positions[barrier_name]
            if self._alternating:
                if idx % 2 == 0:
                    # Even-indexed: attach at the top.
                    top_y = interior_y_start
                    bottom_barrier_y = interior_y_start + h - 1
                else:
                    # Odd-indexed: attach at the bottom.
                    bottom_barrier_y = interior_y_end - 1
                    top_y = bottom_barrier_y - h + 1
            else:
                # Default: extend upward from the bottom row.
                top_y = bottom_y - h + 1
                bottom_barrier_y = bottom_y

            grid[top_y:bottom_barrier_y + 1, x_start:x_start + self._barrier_width] = self._border_object

        # Draw outer border using slicing.
        grid[:interior_y_start, :] = self._border_object   # Top border.
        grid[interior_y_end:, :] = self._border_object      # Bottom border.
        grid[:, :interior_x_start] = self._border_object    # Left border.
        grid[:, interior_x_end:] = self._border_object      # Right border.

        return grid
