from typing import List, Union
import numpy as np
from omegaconf import DictConfig
from mettagrid.config.room.room import Room

class AlternatingBarrierMaze(Room):
    """
    An environment with an outer wall border where:
      - The agent starts at the leftmost cell of the interior's bottom row.
      - A contiguous cluster of three objects—generator, altar, and converter—is placed flush
        against the rightmost end of the interior.
      - In between are a variable number of barrier blocks. Their heights are provided in a list.
        Barriers alternate in attachment:
          - Even-indexed barriers (0, 2, …) are attached at the top (drawn downward from the top edge).
          - Odd-indexed barriers (1, 3, …) are attached at the bottom (drawn upward from the bottom edge).

    The interior's bottom row is divided into two groups:
      Left group: [ agent ] + [ barrier blocks ]
      Right group (cluster): [ generator, altar, converter ]
    The right group is placed flush with the right interior edge.
    """

    def __init__(
        self,
        width: int,
        height: int,
        num_barriers: int = 3,
        barrier_width: int = 1,
        min_barrier_height: int = 1,
        max_barrier_height: int = None,
        barrier_heights: List[int] = None,
        agents: int | DictConfig = 1,
        seed: int | None = None,
        border_width: int = 1,
        border_object: str = "wall",
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._width = width
        self._height = height
        self._barrier_width = barrier_width
        self._agents = agents
        self._rng = np.random.default_rng(seed)
        self._border_width = border_width
        self._border_object = border_object

        # Handle barrier heights
        if max_barrier_height is None:
            max_barrier_height = height - (2 * border_width) - 1

        if barrier_heights is None:
            # Generate random barrier heights if not provided
            self._barrier_heights = [
                self._rng.integers(min_barrier_height, max_barrier_height + 1)
                for _ in range(num_barriers)
            ]
        else:
            self._barrier_heights = barrier_heights[:num_barriers]  # Use only num_barriers heights
            # If barrier_heights is shorter than num_barriers, generate random heights for the rest
            while len(self._barrier_heights) < num_barriers:
                self._barrier_heights.append(
                    self._rng.integers(min_barrier_height, max_barrier_height + 1)
                )

        # Initialize the grid with "empty" cells.
        self._grid = np.full((self._height, self._width), "empty", dtype='<U50')

    def _build(self) -> np.ndarray:
        # Define the interior (area inside the border).
        interior_x_start = self._border_width
        interior_x_end = self._width - self._border_width
        interior_y_start = self._border_width
        interior_y_end = self._height - self._border_width
        interior_width = interior_x_end - interior_x_start
        bottom_y = interior_y_end - 1  # bottom row of the interior

        # --- Define Groups of Blocks ---
        # Left group: agent and one barrier block per barrier height.
        num_barriers = len(self._barrier_heights)
        left_group_blocks = [("agent", 1)] + [("barrier", self._barrier_width) for _ in range(num_barriers)]
        # Right group (cluster): generator, altar, converter.
        right_group_blocks = [("generator", 1), ("altar", 1), ("converter", 1)]
        cluster_width = sum(width for (_, width) in right_group_blocks)
        # Place the cluster flush with the right interior edge.
        groupB_start_x = interior_x_end - cluster_width

        # The left group must occupy the space from interior_x_start up to groupB_start_x.
        available_left_space = groupB_start_x - interior_x_start
        left_total_width = sum(width for (_, width) in left_group_blocks)
        assert available_left_space >= left_total_width, (
            "Interior width is too small for the left group blocks."
        )
        left_available_gap = available_left_space - left_total_width
        num_gaps_left = len(left_group_blocks) - 1 if len(left_group_blocks) > 1 else 0
        gap_left = left_available_gap // num_gaps_left if num_gaps_left > 0 else 0
        remainder_left = left_available_gap % num_gaps_left if num_gaps_left > 0 else 0

        # --- Compute X-Positions for Blocks ---
        x_positions = {}

        # Place left group blocks.
        current_x = interior_x_start
        # Place the agent.
        block_name, block_width = left_group_blocks[0]
        x_positions[block_name] = current_x
        current_x += block_width

        # For barrier blocks, record them as "barrier0", "barrier1", etc.
        left_barrier_indices = []
        for i in range(1, len(left_group_blocks)):
            extra_gap = gap_left + (1 if i <= remainder_left else 0)
            current_x += extra_gap
            name, width_block = left_group_blocks[i]
            if name == "barrier":
                barrier_key = f"barrier{i-1}"
                x_positions[barrier_key] = current_x
                left_barrier_indices.append(barrier_key)
            else:
                x_positions[name] = current_x
            current_x += width_block

        # Place right group blocks contiguously, flush with the right edge.
        current_x = groupB_start_x
        for name, width_block in right_group_blocks:
            x_positions[name] = current_x
            current_x += width_block

        # --- Place Fixed Floor Items on the Bottom Row ---
        self._grid[bottom_y, x_positions["agent"]] = "agent.agent"
        self._grid[bottom_y, x_positions["generator"]] = "generator"
        self._grid[bottom_y, x_positions["altar"]] = "altar"
        self._grid[bottom_y, x_positions["converter"]] = "converter"

        # --- Draw Each Barrier as a Vertical Wall ---
        # Barriers are taken from the left group (keys "barrier0", "barrier1", ...).
        for i, barrier_key in enumerate(left_barrier_indices):
            barrier_x = x_positions[barrier_key]
            barrier_height = self._barrier_heights[i]
            if i % 2 == 0:
                # Even-indexed: attach at the top of the interior.
                top_y = interior_y_start
                bottom_barrier_y = interior_y_start + barrier_height - 1
                for x in range(barrier_x, barrier_x + self._barrier_width):
                    for y in range(top_y, bottom_barrier_y + 1):
                        self._grid[y, x] = self._border_object
            else:
                # Odd-indexed: attach at the bottom of the interior.
                bottom_barrier_y = interior_y_end - 1
                top_y = bottom_barrier_y - barrier_height + 1
                for x in range(barrier_x, barrier_x + self._barrier_width):
                    for y in range(top_y, bottom_barrier_y + 1):
                        self._grid[y, x] = self._border_object

        # --- Draw the Outer Border ---
        # Top border.
        for y in range(0, interior_y_start):
            self._grid[y, :] = self._border_object
        # Bottom border.
        for y in range(interior_y_end, self._height):
            self._grid[y, :] = self._border_object
        # Left border.
        for x in range(0, interior_x_start):
            self._grid[:, x] = self._border_object
        # Right border.
        for x in range(interior_x_end, self._width):
            self._grid[:, x] = self._border_object

        return self._grid
