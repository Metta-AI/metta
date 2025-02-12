from typing import List
import numpy as np
from omegaconf import DictConfig
from mettagrid.config.room.room import Room

class BarrierMaze(Room):
    """
    An environment with a wall border and three vertical barriers along the bottom row of the interior.
    
    The interior (area inside the border) bottom row is partitioned into seven blocks (from left to right):
      1. Agent start ("agent.agent")
      2. Barrier 1 (vertical wall extending upward for a configurable height)
      3. Generator ("generator")
      4. Barrier 2 (vertical wall)
      5. Converter ("converter")
      6. Barrier 3 (vertical wall)
      7. Altar ("altar")
    
    Any extra horizontal space is evenly distributed as gaps between blocks so that the agent appears at
    the left edge and the altar at the right edge of the interior.
    """
    def __init__(
        self,
        width: int,
        height: int,
        barrier_heights: List[int],
        barrier_width: int = 1,
        agents: int | DictConfig = 1,
        seed: int | None = None,
        border_width: int = 1,
        border_object: str = "wall",
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._width = width
        self._height = height
        self._barrier_heights = barrier_heights
        self._barrier_width = barrier_width
        self._agents = agents
        self._rng = np.random.default_rng(seed)
        self._border_width = border_width
        self._border_object = border_object

        # Initialize the grid with "empty" cells.
        self._grid = np.full((self._height, self._width), "empty", dtype='<U50')

    def _build(self) -> np.ndarray:
        # Define the interior boundaries (area inside the border).
        interior_x_start = self._border_width
        interior_x_end = self._width - self._border_width
        interior_y_start = self._border_width
        interior_y_end = self._height - self._border_width

        interior_width = interior_x_end - interior_x_start
        # (We don’t need interior_height here except to compute the bottom row.)
        bottom_y = interior_y_end - 1  # The bottom row of the interior.

        # Define the blocks along the bottom row in order:
        # ("name", width). Non-barrier items are 1 cell wide;
        # barriers use the configured barrier_width.
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

        # Distribute the gap space evenly between the blocks.
        num_gaps = len(blocks) - 1
        base_gap = total_gap // num_gaps if num_gaps > 0 else 0
        remainder = total_gap % num_gaps if num_gaps > 0 else 0

        # Compute the starting x-coordinate (within the interior) for each block.
        x_positions = {}
        current_x = interior_x_start
        # Place the first block: agent.
        block_name, block_width = blocks[0]
        x_positions[block_name] = current_x
        current_x += block_width

        for i in range(1, len(blocks)):
            gap = base_gap + (1 if i <= remainder else 0)
            current_x += gap
            block_name, block_width = blocks[i]
            x_positions[block_name] = current_x
            current_x += block_width

        # Place the floor items (agent, generator, converter, altar) on the interior's bottom row.
        self._grid[bottom_y, x_positions["agent"]] = "agent.agent"
        self._grid[bottom_y, x_positions["generator"]] = "generator"
        self._grid[bottom_y, x_positions["converter"]] = "converter"
        self._grid[bottom_y, x_positions["altar"]] = "altar"

        # Draw each barrier as a vertical wall that extends upward from the bottom row.
        # The wall for each barrier will extend for the configured height.
        # Barrier 1:
        b1_height = self._barrier_heights[0]
        top_y_b1 = bottom_y - b1_height + 1
        for x in range(x_positions["barrier1"], x_positions["barrier1"] + self._barrier_width):
            for y in range(top_y_b1, bottom_y + 1):
                self._grid[y, x] = self._border_object  # using "wall" as the barrier symbol

        # Barrier 2:
        b2_height = self._barrier_heights[1]
        top_y_b2 = bottom_y - b2_height + 1
        for x in range(x_positions["barrier2"], x_positions["barrier2"] + self._barrier_width):
            for y in range(top_y_b2, bottom_y + 1):
                self._grid[y, x] = self._border_object

        # Barrier 3:
        b3_height = self._barrier_heights[2]
        top_y_b3 = bottom_y - b3_height + 1
        for x in range(x_positions["barrier3"], x_positions["barrier3"] + self._barrier_width):
            for y in range(top_y_b3, bottom_y + 1):
                self._grid[y, x] = self._border_object

        # Finally, add the outer border around the entire environment.
        # Top border:
        for y in range(0, interior_y_start):
            self._grid[y, :] = self._border_object
        # Bottom border:
        for y in range(interior_y_end, self._height):
            self._grid[y, :] = self._border_object
        # Left border:
        for x in range(0, interior_x_start):
            self._grid[:, x] = self._border_object
        # Right border:
        for x in range(interior_x_end, self._width):
            self._grid[:, x] = self._border_object

        return self._grid
