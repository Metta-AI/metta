"""
MinGrid-Inspired Crossing Paths Environment

Inspired by MinGrid's CrossingEnv. This creates an environment with intersecting corridors
where agents must navigate through crossing paths to reach objectives. The challenge is
navigating efficiently through the corridor intersections.

Original MinGrid CrossingEnv: Agent navigates through intersecting corridors to reach goal.
MettagGrid adaptation: Agent navigates intersecting corridors to reach altars/generators.

The environment consists of:
- Intersecting horizontal and vertical corridors
- Walls forming the corridor structure
- Altars/generators placed at corridor ends or intersections
- Agents must choose correct paths through intersections
"""

from typing import Optional, Tuple, List
import numpy as np
from omegaconf import DictConfig

from mettagrid.room.room import Room


class MinigridCrossingPaths(Room):
    def __init__(
        self,
        width: int,
        height: int,
        objects: DictConfig,
        agents: int | DictConfig = 1,
        seed: Optional[int] = None,
        border_width: int = 1,
        border_object: str = "wall",
        corridor_width: int = 3,  # Width of corridors
        num_horizontal_corridors: int = 2,  # Number of horizontal corridors
        num_vertical_corridors: int = 2,  # Number of vertical corridors
        team: str | None = None,
    ):
        super().__init__(border_width=border_width, border_object=border_object, labels=["minigrid_crossing_paths"])
        self.set_size_labels(width, height)
        self._rng = np.random.default_rng(seed)
        self._width = width
        self._height = height
        self._agents = agents
        self._objects = objects
        self._corridor_width = corridor_width
        self._num_horizontal_corridors = num_horizontal_corridors
        self._num_vertical_corridors = num_vertical_corridors
        self._team = team

    def _build(self) -> np.ndarray:
        # Start with all walls
        grid = np.full((self._height, self._width), "wall", dtype="<U50")

        # Create corridor structure
        corridor_info = self._create_corridors(grid)

        # Place resources at strategic locations
        self._place_resources_in_corridors(grid, corridor_info)

        # Place agents at corridor entrance
        self._place_agents_in_corridors(grid, corridor_info)

        return grid

    def _create_corridors(self, grid: np.ndarray) -> dict:
        """Create intersecting horizontal and vertical corridors."""
        corridor_info = {
            "horizontal_corridors": [],
            "vertical_corridors": [],
            "intersections": []
        }

        # Create horizontal corridors
        available_height = self._height - 2  # Account for borders
        h_spacing = available_height // (self._num_horizontal_corridors + 1)

        for i in range(self._num_horizontal_corridors):
            corridor_y = (i + 1) * h_spacing
            corridor_top = max(1, corridor_y - self._corridor_width // 2)
            corridor_bottom = min(self._height - 1, corridor_y + self._corridor_width // 2 + 1)

            # Clear horizontal corridor
            for r in range(corridor_top, corridor_bottom):
                for c in range(1, self._width - 1):
                    grid[r, c] = "empty"

            corridor_info["horizontal_corridors"].append((corridor_top, corridor_bottom, 1, self._width - 1))

        # Create vertical corridors
        available_width = self._width - 2  # Account for borders
        v_spacing = available_width // (self._num_vertical_corridors + 1)

        for i in range(self._num_vertical_corridors):
            corridor_x = (i + 1) * v_spacing
            corridor_left = max(1, corridor_x - self._corridor_width // 2)
            corridor_right = min(self._width - 1, corridor_x + self._corridor_width // 2 + 1)

            # Clear vertical corridor
            for r in range(1, self._height - 1):
                for c in range(corridor_left, corridor_right):
                    grid[r, c] = "empty"

            corridor_info["vertical_corridors"].append((1, self._height - 1, corridor_left, corridor_right))

        # Find intersections
        for h_top, h_bottom, h_left, h_right in corridor_info["horizontal_corridors"]:
            for v_top, v_bottom, v_left, v_right in corridor_info["vertical_corridors"]:
                # Check if corridors intersect
                if (h_top < v_bottom and h_bottom > v_top and
                    h_left < v_right and h_right > v_left):

                    intersection = (
                        max(h_top, v_top),      # top
                        max(h_left, v_left),    # left
                        min(h_bottom, v_bottom), # bottom
                        min(h_right, v_right)   # right
                    )
                    corridor_info["intersections"].append(intersection)

        return corridor_info

    def _place_resources_in_corridors(self, grid: np.ndarray, corridor_info: dict) -> None:
        """Place resources at corridor ends and intersections."""
        # Collect strategic positions
        strategic_positions = []

        # Add corridor end positions
        for h_top, h_bottom, h_left, h_right in corridor_info["horizontal_corridors"]:
            # Left end
            center_r = (h_top + h_bottom) // 2
            strategic_positions.append((center_r, h_left + 1))
            # Right end
            strategic_positions.append((center_r, h_right - 2))

        for v_top, v_bottom, v_left, v_right in corridor_info["vertical_corridors"]:
            # Top end
            center_c = (v_left + v_right) // 2
            strategic_positions.append((v_top + 1, center_c))
            # Bottom end
            strategic_positions.append((v_bottom - 2, center_c))

        # Add intersection centers
        for int_top, int_left, int_bottom, int_right in corridor_info["intersections"]:
            center_r = (int_top + int_bottom) // 2
            center_c = (int_left + int_right) // 2
            strategic_positions.append((center_r, center_c))

        # Filter valid positions
        valid_positions = []
        for r, c in strategic_positions:
            if (0 <= r < self._height and 0 <= c < self._width and
                grid[r, c] == "empty"):
                valid_positions.append((r, c))

        # Place altars
        altar_count = self._objects.get("altar", 0)
        altar_positions = self._select_positions(valid_positions, altar_count)
        for r, c in altar_positions:
            grid[r, c] = "altar"
            valid_positions.remove((r, c))

        # Place generators
        generator_count = self._objects.get("generator", 0)
        generator_positions = self._select_positions(valid_positions, generator_count)
        for r, c in generator_positions:
            grid[r, c] = "generator"
            valid_positions.remove((r, c))

        # Place other objects
        for obj_name, obj_count in self._objects.items():
            if obj_name in ["altar", "generator"]:
                continue

            positions = self._select_positions(valid_positions, obj_count)
            for r, c in positions:
                grid[r, c] = obj_name
                if (r, c) in valid_positions:
                    valid_positions.remove((r, c))

    def _select_positions(self, available_positions: List[Tuple[int, int]], count: int) -> List[Tuple[int, int]]:
        """Select positions for placing objects."""
        if not available_positions or count <= 0:
            return []

        selected_count = min(count, len(available_positions))
        if selected_count == len(available_positions):
            return available_positions.copy()

        # Randomly select positions
        indices = self._rng.choice(len(available_positions), size=selected_count, replace=False)
        return [available_positions[i] for i in indices]

    def _place_agents_in_corridors(self, grid: np.ndarray, corridor_info: dict) -> None:
        """Place agents at corridor entrances."""
        # Find good starting positions (corridor entrances)
        start_positions = []

        # Use horizontal corridor entrances
        for h_top, h_bottom, h_left, h_right in corridor_info["horizontal_corridors"]:
            center_r = (h_top + h_bottom) // 2
            # Left entrance
            if h_left + 2 < self._width and grid[center_r, h_left + 2] == "empty":
                start_positions.append((center_r, h_left + 2))

        # Use vertical corridor entrances if needed
        if len(start_positions) < self._agents:
            for v_top, v_bottom, v_left, v_right in corridor_info["vertical_corridors"]:
                center_c = (v_left + v_right) // 2
                # Top entrance
                if v_top + 2 < self._height and grid[v_top + 2, center_c] == "empty":
                    start_positions.append((v_top + 2, center_c))

        # If still need more positions, use any empty corridor space
        if len(start_positions) < self._agents:
            for r in range(self._height):
                for c in range(self._width):
                    if grid[r, c] == "empty" and len(start_positions) < self._agents:
                        start_positions.append((r, c))

        # Place agents
        if isinstance(self._agents, int):
            if self._team is None:
                agents = ["agent.agent"] * self._agents
            else:
                agents = [f"agent.{self._team}"] * self._agents
        elif isinstance(self._agents, DictConfig):
            agents = [f"agent.{agent}" for agent, na in self._agents.items() for _ in range(na)]

        for i, agent in enumerate(agents):
            if i < len(start_positions):
                r, c = start_positions[i]
                grid[r, c] = agent
