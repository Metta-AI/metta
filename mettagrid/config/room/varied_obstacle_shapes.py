from typing import Optional
import numpy as np
from omegaconf import DictConfig

from mettagrid.config.room.room import Room

class VariedObstacleShapes(Room):
    def __init__(
        self,
        width: int,
        height: int,
        objects: DictConfig,
        agents: int | DictConfig = 0,
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: str = "wall"
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._rng = np.random.default_rng(seed)
        self._width = width
        self._height = height
        self._objects = objects
        self._agents = agents

        # Define a list of obstacle patterns with varying numbers of wall blocks (from 1 up to 7)
        self.obstacle_patterns = [
            np.array([["wall"]]),  # 1 block
            np.array([["wall", "wall"]]),  # 2 blocks (horizontal line)
            np.array([["wall", "wall", "wall"]]),  # 3 blocks (horizontal line)
            np.array([["wall", "wall"],
                      ["wall", "wall"]]),  # 4 blocks (2x2 square)
            np.array([["empty", "wall", "empty"],
                      ["wall", "wall", "wall"],
                      ["empty", "wall", "empty"]]),  # 5 blocks (cross)
            np.array([["wall", "wall", "wall"],
                      ["wall", "empty", "empty"],
                      ["wall", "wall", "empty"]]),  # 6 blocks (L-shaped variant)
            np.array([["wall", "wall", "empty"],
                      ["empty", "wall", "wall"],
                      ["wall", "wall", "wall"]])  # 7 blocks (zigzag)
        ]

    def _build(self):
        # Prepare agent symbols
        if isinstance(self._agents, int):
            agents = ["agent.agent"] * self._agents
        elif isinstance(self._agents, DictConfig):
            agents = ["agent." + agent for agent, na in self._agents.items() for _ in range(na)]
        else:
            agents = []

        # Adjust object counts if total exceeds 2/3 of room area
        total_objects = sum(self._objects.values()) + len(agents)
        area = self._width * self._height
        while total_objects > 2 * area / 3:
            for obj_name in self._objects:
                self._objects[obj_name] = max(1, self._objects[obj_name] // 2)
            total_objects = sum(self._objects.values()) + len(agents)

        # Create an empty grid
        grid = np.full((self._height, self._width), "empty", dtype=object)

        # Use the "wall" count to determine the number of obstacles to place
        obstacle_count = self._objects.get("wall", 0)
        grid = self._place_obstacles(grid, obstacle_count)

        # Place remaining objects (excluding walls) in random empty cells
        for obj, count in self._objects.items():
            if obj == "wall":
                continue
            for _ in range(count):
                empty_positions = np.argwhere(grid == "empty")
                if len(empty_positions) == 0:
                    raise ValueError("No empty space available for object placement.")
                pos_idx = self._rng.integers(0, len(empty_positions))
                pos = empty_positions[pos_idx]
                grid[pos[0], pos[1]] = obj

        # Place agents in remaining empty cells
        for agent in agents:
            empty_positions = np.argwhere(grid == "empty")
            if len(empty_positions) == 0:
                raise ValueError("No empty space available for agent placement.")
            pos_idx = self._rng.integers(0, len(empty_positions))
            pos = empty_positions[pos_idx]
            grid[pos[0], pos[1]] = agent

        return grid

    def _place_obstacles(self, grid, obstacle_count: int):
        """
        Places composite obstacle objects in the grid. For each obstacle, a pattern is chosen randomly from a list
        of shapes (each with a different number of wall blocks, from 1 to 7). The pattern is then placed with a one-cell
        clearance around it to ensure traversability.
        """
        placed = 0
        attempts = 0
        height, width = grid.shape

        while placed < obstacle_count and attempts < obstacle_count * 100:
            attempts += 1
            # Randomly select one of the obstacle patterns
            pattern = self.obstacle_patterns[self._rng.integers(0, len(self.obstacle_patterns))]
            p_h, p_w = pattern.shape

            # Set clearance (one cell on each side)
            clearance = 1
            effective_h = p_h + 2 * clearance
            effective_w = p_w + 2 * clearance

            if height - effective_h < 0 or width - effective_w < 0:
                break

            # Choose a random top-left coordinate for the effective region
            r = self._rng.integers(0, height - effective_h + 1)
            c = self._rng.integers(0, width - effective_w + 1)

            # Check if the effective region is completely empty
            region = grid[r:r+effective_h, c:c+effective_w]
            if np.all(region == "empty"):
                # Place the pattern in the center of the effective region
                start_r = r + clearance
                start_c = c + clearance
                grid[start_r:start_r+p_h, start_c:start_c+p_w] = pattern
                placed += 1

        if placed < obstacle_count:
            print(f"Warning: Only placed {placed} out of {obstacle_count} obstacles after {attempts} attempts.")

        return grid