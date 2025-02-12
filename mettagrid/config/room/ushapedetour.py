from typing import Set, Tuple
import numpy as np
from omegaconf import DictConfig
from mettagrid.config.room.room import Room

class UShapedetour(Room):
    """
    A U-shaped detour maze where resource objects are placed behind a barrier relative to the agent.

    The agent starts in the top region of the maze. A horizontal wall barrier is placed below the agent
    with a gap that allows passage. Reward objects (a heart altar, a generator, and a converter)
    are scattered in the bottom region (behind the barrier), so that they are initially out of direct sight.
    """

    def __init__(
        self,
        width: int,
        height: int,
        detour_params: DictConfig,
        agents: int = 1,
        seed: int | None = None,
        border_width: int = 1,
        border_object: str = "wall",
    ):
        """
        Parameters:
          width (int): Total number of columns in the grid.
          height (int): Total number of rows in the grid.
          detour_params (DictConfig): Parameters for the barrier.
            Expected keys:
              - barrier_row: The y-coordinate where the barrier is placed.
              - barrier_thickness: How many rows the barrier spans.
              - barrier_gap_width: The width (in cells) of the gap in the barrier.
          agents (int): Number of agents (should be 1).
          seed (int, optional): Seed for randomness.
          border_width (int): Width of the outer border.
          border_object (str): The object used for the border (e.g., "wall").
        """
        super().__init__(border_width=border_width, border_object=border_object)
        self._width = width
        self._height = height
        self._detour_params = detour_params
        self._agents = agents
        self._rng = np.random.default_rng(seed)
        self._border_width = border_width

        # Initialize grid with "empty" cells.
        self._grid = np.full((self._height, self._width), "empty", dtype='<U50')
        # Track positions that are occupied.
        self._placed_positions: Set[Tuple[int, int]] = set()

        # Place the border around the edges.
        for y in range(self._height):
            for x in range(self._width):
                if (x < self._border_width or x >= self._width - self._border_width or
                    y < self._border_width or y >= self._height - self._border_width):
                    self._grid[y, x] = border_object
                    self._placed_positions.add((x, y))

    def _build(self) -> np.ndarray:
        """
        Build the U-shaped detour maze:
          1. Place a horizontal barrier (with a gap) at a specified row.
          2. Place the agent in the top region (above the barrier).
          3. Scatter one heart altar ("altar"), one generator ("generator"),
             and one converter ("converter") in the bottom region (below the barrier).
        """
        # Extract barrier parameters.
        barrier_row = self._detour_params.get("barrier_row", int(self._height * 0.3))
        barrier_thickness = self._detour_params.get("barrier_thickness", 1)
        barrier_gap_width = self._detour_params.get("barrier_gap_width", 3)

        # Define the interior boundaries (excluding the borders).
        interior_x_start = self._border_width
        interior_x_end = self._width - self._border_width  # non-inclusive

        # Create the horizontal barrier rows.
        for row in range(barrier_row, barrier_row + barrier_thickness):
            # Randomly choose a gap starting position so the gap fits.
            gap_start = int(self._rng.integers(interior_x_start, interior_x_end - barrier_gap_width + 1))
            for x in range(interior_x_start, interior_x_end):
                # Place a wall unless x is in the gap.
                if x < gap_start or x >= gap_start + barrier_gap_width:
                    self._grid[row, x] = "wall"
                    self._placed_positions.add((x, row))

        # --- Place the agent in the top region ---
        agent_region_y_min = self._border_width
        agent_region_y_max = barrier_row - 1  # Agent spawns above the barrier.
        agent_position = self._sample_position(
            interior_x_start, interior_x_end - 1, agent_region_y_min, agent_region_y_max
        )
        self._grid[agent_position[1], agent_position[0]] = "agent.agent"
        self._placed_positions.add(agent_position)

        # --- Place resource objects in the bottom region ---
        resource_region_y_min = barrier_row + barrier_thickness
        resource_region_y_max = self._height - self._border_width - 1
        for obj, symbol in zip(["heart_altar", "generator", "converter"],
                                 ["altar", "generator", "converter"]):
            pos = self._sample_position(
                interior_x_start, interior_x_end - 1, resource_region_y_min, resource_region_y_max
            )
            self._grid[pos[1], pos[0]] = symbol
            self._placed_positions.add(pos)

        return self._grid

    def _sample_position(
        self, x_min: int, x_max: int, y_min: int, y_max: int
    ) -> Tuple[int, int]:
        """
        Sample a random (x, y) position within the given inclusive bounds that is not already occupied.
        """
        for _ in range(100):
            x = int(self._rng.integers(x_min, x_max + 1))
            y = int(self._rng.integers(y_min, y_max + 1))
            pos = (x, y)
            if pos not in self._placed_positions:
                return pos
        # Fallback if no free position is found.
        return (x_min, y_min)
