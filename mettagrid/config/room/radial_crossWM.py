from typing import Set, Tuple
import numpy as np
from omegaconf import DictConfig
from mettagrid.config.room.room import Room

class RadialCross(Room):
    """
    A cross-shaped maze with four arms. In this variant:
      - The top arm contains the agent and, next to it, the heart altar ("altar").
      - The right arm has the generator.
      - The left arm has the converter.
      - The bottom arm is left empty.

    The overall maze dimensions and the arm width are fully parameterizable.
    """

    def __init__(
        self,
        width: int,
        height: int,
        arm_width: int,
        agents: int = 1,
        seed: int | None = None,
        border_width: int = 1,
        border_object: str = "wall",
    ):
        """
        Parameters:
          width (int): Total number of columns in the grid.
          height (int): Total number of rows in the grid.
          arm_width (int): The thickness of each arm in the cross.
          agents (int): Number of agents (should be 1 for this environment).
          seed (int, optional): Seed for randomness.
          border_width (int): The width of the fixed border.
          border_object (str): The object used for the border (e.g. "wall").
        """
        super().__init__(border_width=border_width, border_object=border_object)
        self._width = width
        self._height = height
        self._arm_width = arm_width
        self._agents = agents
        self._rng = np.random.default_rng(seed)

        # Compute interior dimensions (area not occupied by the border)
        interior_width = self._width - 2 * border_width
        interior_height = self._height - 2 * border_width
        assert interior_width > 0 and interior_height > 0, "Maze too small for the given border width."
        assert arm_width <= interior_width and arm_width <= interior_height, (
            "arm_width is too large for the given maze dimensions."
        )

        # Initialize the grid filled with the border object.
        self._grid = np.full((self._height, self._width), border_object, dtype='<U50')

    def _build(self) -> np.ndarray:
        # Define interior boundaries (leaving the border intact).
        interior_x_start = self._border_width
        interior_x_end = self._width - self._border_width
        interior_y_start = self._border_width
        interior_y_end = self._height - self._border_width

        # Compute the center of the interior.
        center_x = (interior_x_start + interior_x_end) // 2
        center_y = (interior_y_start + interior_y_end) // 2

        # Calculate half the arm thickness.
        half_arm = self._arm_width // 2

        # Carve the horizontal corridor (from center_y - half_arm to center_y + half_arm).
        for r in range(center_y - half_arm, center_y + half_arm + 1):
            for c in range(interior_x_start, interior_x_end):
                self._grid[r, c] = "empty"

        # Carve the vertical corridor (from center_x - half_arm to center_x + half_arm).
        for c in range(center_x - half_arm, center_x + half_arm + 1):
            for r in range(interior_y_start, interior_y_end):
                self._grid[r, c] = "empty"

        # --- Object Placement ---

        # Right arm: Place the generator.
        right_col = interior_x_end - 1
        self._grid[center_y, right_col] = "generator"

        # Left arm: Place the converter.
        left_col = interior_x_start
        self._grid[center_y, left_col] = "converter"

        # Top arm: Carve out the top row and place the agent and the heart altar.
        top_row = interior_y_start
        # Ensure the entire top row is empty so the objects can be placed there.
        for c in range(interior_x_start, interior_x_end):
            self._grid[top_row, c] = "empty"

        # Place the agent at the top-center, and the heart altar immediately to its right (if possible).
        agent_col = center_x
        if agent_col + 1 < interior_x_end:
            altar_col = agent_col + 1
        else:
            altar_col = agent_col - 1  # fallback if there's no room to the right

        self._grid[top_row, agent_col] = "agent.agent"
        self._grid[top_row, altar_col] = "altar"

        return self._grid
