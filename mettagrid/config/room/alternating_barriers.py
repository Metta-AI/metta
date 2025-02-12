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

        # Compute the maximum possible barrier height.
        max_barrier_height = height - (2 * border_width) - 1

        # Generate random barrier heights if none are provided.
        if barrier_heights is None:
            self._barrier_heights = [
                self._rng.integers(1, max_barrier_height + 1) for _ in range(num_barriers)
            ]
        else:
            assert len(barrier_heights) == num_barriers, "Barrier heights must match the number of barriers."
            self._barrier_heights = [min(h, max_barrier_height) for h in barrier_heights]

    def _build(self) -> np.ndarray:
        grid = np.full((self._height, self._width), "empty", dtype='<U50')
        # Define interior boundaries.
        bw = self._border_width
        interior_x_start, interior_x_end = bw, self._width - bw
        interior_y_start, interior_y_end = bw, self._height - bw
        bottom_y = interior_y_end - 1

        # --- Define Block Groups ---
        # Left group: agent followed by barrier blocks.
        left_blocks = [("agent", 1)] + [("barrier", self._barrier_width) for _ in range(len(self._barrier_heights))]
        # Right group (cluster): generator, altar, and converter.
        right_blocks = [("generator", 1), ("altar", 1), ("converter", 1)]
        cluster_width = sum(width for _, width in right_blocks)
        groupB_start_x = interior_x_end - cluster_width

        # --- Compute X-Positions for Blocks ---
        # Calculate available space for the left group.
        available_space = groupB_start_x - interior_x_start
        left_total_width = sum(width for _, width in left_blocks)
        assert available_space >= left_total_width, "Interior width is too small for the left group blocks."

        total_gap = available_space - left_total_width
        num_gaps = len(left_blocks) - 1 if len(left_blocks) > 1 else 0
        gap, remainder = (total_gap // num_gaps, total_gap % num_gaps) if num_gaps else (0, 0)

        x_positions = {}
        current_x = interior_x_start

        # Place the agent.
        x_positions["agent"] = current_x
        current_x += 1

        # Place barrier blocks with evenly distributed gaps.
        barrier_keys = []
        for i in range(1, len(left_blocks)):
            extra_gap = gap + (1 if i <= remainder else 0)
            current_x += extra_gap
            barrier_key = f"barrier{i-1}"
            x_positions[barrier_key] = current_x
            barrier_keys.append(barrier_key)
            current_x += self._barrier_width

        # Place right group blocks contiguously.
        current_x = groupB_start_x
        for name, width in right_blocks:
            x_positions[name] = current_x
            current_x += width

        # --- Place Fixed Items on the Bottom Row ---
        placements = {
            "agent": "agent.agent",
            "generator": "generator",
            "altar": "altar",
            "converter": "converter",
        }
        for key, value in placements.items():
            grid[bottom_y, x_positions[key]] = value

        # --- Draw Barriers ---
        for i, key in enumerate(barrier_keys):
            barrier_x = x_positions[key]
            barrier_height = self._barrier_heights[i]
            if i % 2 == 0:
                # Even-indexed barrier: attach at the top.
                grid[interior_y_start: interior_y_start + barrier_height,
                           barrier_x: barrier_x + self._barrier_width] = self._border_object
            else:
                # Odd-indexed barrier: attach at the bottom.
                grid[interior_y_end - barrier_height: interior_y_end,
                           barrier_x: barrier_x + self._barrier_width] = self._border_object

        # --- Draw Outer Border using slicing ---
        grid[:interior_y_start, :] = self._border_object  # Top border.
        grid[interior_y_end:, :] = self._border_object     # Bottom border.
        grid[:, :interior_x_start] = self._border_object   # Left border.
        grid[:, interior_x_end:] = self._border_object     # Right border.

        return grid
