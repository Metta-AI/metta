from typing import Set, Tuple
import numpy as np
from omegaconf import DictConfig
from mettagrid.config.room.room import Room

class Grok(Room):
    """
    The Grok maze is designed with two distinct regions:
      - Near region: The agent spawns in the top-left corner (just inside the border).
        Close to the agent is a small, fully enclosed rectangular area that contains one or more generators.
        Because the enclosure is completely sealed by walls, these generators are inaccessible.
      - Far region: In the bottom-right portion of the maze, a heart altar ("altar"),
        a generator ("generator"), and a converter ("converter") are scattered in the open.
        These objects are accessible to the agent.
    """

    def __init__(
        self,
        width: int,
        height: int,
        grok_params: DictConfig,
        agents: int = 1,
        seed: int | None = None,
        border_width: int = 1,
        border_object: str = "wall",
    ):
        """
        Parameters:
          width (int): Total number of columns in the grid.
          height (int): Total number of rows in the grid.
          grok_params (DictConfig): Parameters for configuring the Grok maze.
            Expected keys:
              - near_enclosure_width: The width of the near enclosure.
              - near_enclosure_height: The height of the near enclosure.
              - near_generator_count: The number of generators to place inside the near enclosure.
          agents (int): Number of agents (should be 1 for this room).
          seed (int, optional): Seed for randomness.
          border_width (int): The thickness of the outer border.
          border_object (str): The object used for the border (e.g., "wall").
        """
        super().__init__(border_width=border_width, border_object=border_object)
        self._width = width
        self._height = height
        self._grok_params = grok_params
        self._agents = agents
        self._rng = np.random.default_rng(seed)
        self._border_width = border_width

        # Initialize grid with "empty" cells.
        self._grid = np.full((self._height, self._width), "empty", dtype='<U50')
        # Keep track of positions that are occupied.
        self._placed_positions: Set[Tuple[int, int]] = set()

        # Place the outer border.
        for y in range(self._height):
            for x in range(self._width):
                if (x < border_width or x >= self._width - border_width or
                    y < border_width or y >= self._height - border_width):
                    self._grid[y, x] = border_object
                    self._placed_positions.add((x, y))

    def _build(self) -> np.ndarray:
        """
        Build the Grok maze by performing the following steps:
          1. Place the agent in the top-left corner (inside the border).
          2. In the near region (top-left), build an enclosure by drawing a complete
             wall boundary, then place a number of generators inside it.
          3. In the far region (bottom-right), scatter one heart altar ("altar"),
             one generator ("generator"), and one converter ("converter") in open space.
        Returns:
          The completed grid (numpy.ndarray) with object strings.
        """
        # --- Step 1. Place the Agent ---
        # Place the agent just inside the border at the top-left.
        agent_position = (self._border_width, self._border_width)
        self._grid[agent_position[1], agent_position[0]] = "agent.agent"
        self._placed_positions.add(agent_position)

        # --- Step 2. Build the Near Region Enclosure ---
        # Define the enclosure coordinates (placed near the top-left, but not overlapping the agent).
        enclosure_x0 = self._border_width + 1
        enclosure_y0 = self._border_width + 1
        near_enclosure_width = self._grok_params.get("near_enclosure_width", 6)
        near_enclosure_height = self._grok_params.get("near_enclosure_height", 6)
        # Determine the inclusive boundaries of the enclosure.
        enclosure_x1 = enclosure_x0 + near_enclosure_width - 1
        enclosure_y1 = enclosure_y0 + near_enclosure_height - 1

        # Draw the enclosure's wall border (all around the rectangle).
        for y in range(enclosure_y0, enclosure_y1 + 1):
            for x in range(enclosure_x0, enclosure_x1 + 1):
                if x == enclosure_x0 or x == enclosure_x1 or y == enclosure_y0 or y == enclosure_y1:
                    self._grid[y, x] = "wall"
                    self._placed_positions.add((x, y))
        # Place near generators inside the enclosure.
        near_generator_count = self._grok_params.get("near_generator_count", 2)
        # The interior of the enclosure (where generators can be placed) excludes the border.
        for _ in range(near_generator_count):
            pos = self._sample_position(
                enclosure_x0 + 1, enclosure_x1 - 1,
                enclosure_y0 + 1, enclosure_y1 - 1
            )
            self._grid[pos[1], pos[0]] = "generator"
            self._placed_positions.add(pos)

        # --- Step 3. Place Far Region Objects ---
        # Define the far region as the bottom-right area (inside the border).
        far_region_x_min = self._width // 2
        far_region_x_max = self._width - self._border_width - 1
        far_region_y_min = self._height // 2
        far_region_y_max = self._height - self._border_width - 1

        # List the open (accessible) objects to scatter.
        for obj, symbol in zip(
            ["heart_altar", "generator", "converter"],
            ["altar", "generator", "converter"]
        ):
            pos = self._sample_position(
                far_region_x_min, far_region_x_max,
                far_region_y_min, far_region_y_max
            )
            self._grid[pos[1], pos[0]] = symbol
            self._placed_positions.add(pos)

        return self._grid

    def _sample_position(
        self, x_min: int, x_max: int, y_min: int, y_max: int
    ) -> Tuple[int, int]:
        """
        Sample a random (x, y) position within the inclusive bounds [x_min, x_max] and [y_min, y_max]
        that is not already occupied.
        """
        for _ in range(100):
            x = int(self._rng.integers(x_min, x_max + 1))
            y = int(self._rng.integers(y_min, y_max + 1))
            pos = (x, y)
            if pos not in self._placed_positions:
                return pos
        # Fallback if no free position is found.
        return (x_min, y_min)
