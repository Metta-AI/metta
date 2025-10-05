from typing import Optional

import numpy as np

from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from mettagrid.map_builder.utils import draw_border


class AssemblerMapBuilder(MapBuilder):
    class Config(MapBuilderConfig["AssemblerMapBuilder"]):
        seed: Optional[int] = None
        perimeter_objects: dict[str, int] = {}
        center_objects: dict[str, int] = {}
        agents: int | dict[str, int] = 4
        border_width: int = 1
        border_object: str = "wall"
        size: int = 7
        random_scatter: bool = False

    def __init__(self, config: Config):
        self._config = config
        self._rng = np.random.default_rng(self._config.seed)

    def build(self) -> GameMap:
        """Build a map that guarantees all objects are placed."""
        if self._config.seed is not None:
            self._rng = np.random.default_rng(self._config.seed)

        perimeter_list = self._create_object_list(self._config.perimeter_objects)
        center_list = self._create_object_list(self._config.center_objects)
        agent_list = self._create_agent_list()

        # Calculate size based on mode
        if self._config.random_scatter:
            size = self._calculate_size_with_scatter(len(perimeter_list), len(center_list))
        else:
            size = self._calculate_size_structured(len(perimeter_list), len(center_list))

        grid = np.full((size, size), "empty", dtype="<U50")
        draw_border(grid, self._config.border_width, self._config.border_object)

        # Place center objects first (always centered)
        if center_list:
            center_positions = self._get_centered_object_positions(grid, len(center_list))
            self._place_objects(grid, center_list, center_positions)

        # Place perimeter objects (either on perimeter or scattered)
        if perimeter_list:
            if self._config.random_scatter:
                # Get already occupied positions to avoid
                occupied = self._get_occupied_positions(grid)
                self._scatter_objects(grid, perimeter_list, occupied)
            else:
                perimeter_positions = self._get_perimeter_positions(grid, len(perimeter_list))
                self._place_objects(grid, perimeter_list, perimeter_positions)

        # Place agents in remaining empty space
        if agent_list:
            self._place_agents(grid, agent_list)

        return GameMap(grid)

    def _calculate_size_structured(self, num_perimeter: int, num_center: int) -> int:
        """Calculate size for structured placement."""
        import math

        bw = self._config.border_width

        if num_perimeter == 0 and num_center == 0:
            # Just agents, minimal room
            num_agents = self._get_num_agents()
            min_dim = math.ceil(math.sqrt(num_agents))
            return 2 * (bw + 1) + min_dim

        # Calculate space needed for center objects
        if num_center > 0:
            objects_per_side = math.ceil(math.sqrt(num_center))
            center_dim = 2 * (objects_per_side - 1) + 1
        else:
            center_dim = 0

        # Calculate space needed for perimeter
        if num_perimeter > 0:
            # Perimeter needs at least 3 cells per object along the perimeter
            perimeter_inner_dim = math.ceil((num_perimeter * 3 + 4) / 4)
        else:
            perimeter_inner_dim = 0

        # Choose the larger requirement, with extra space if both exist
        if num_perimeter > 0 and num_center > 0:
            inner_dim = max(perimeter_inner_dim, center_dim + 4)
        else:
            inner_dim = max(perimeter_inner_dim, center_dim)

        return 2 * (bw + 1) + inner_dim

    def _calculate_size_with_scatter(self, num_perimeter: int, num_center: int) -> int:
        """Calculate size when perimeter objects will be scattered."""
        import math

        bw = self._config.border_width

        # Calculate space for center objects (same as structured)
        if num_center > 0:
            objects_per_side = math.ceil(math.sqrt(num_center))
            center_dim = 2 * (objects_per_side - 1) + 1
        else:
            center_dim = 0

        # Calculate total objects that need spacing
        total_objects = num_perimeter + num_center

        if total_objects == 0:
            # Just agents
            num_agents = self._get_num_agents()
            min_dim = math.ceil(math.sqrt(num_agents))
            return 2 * (bw + 1) + min_dim

        # Need enough space for all objects with 2-cell spacing
        # Approximate by arranging in a grid
        total_per_side = math.ceil(math.sqrt(total_objects))
        scatter_dim = 2 * total_per_side + 1

        # Take the max of center requirements and total scatter requirements
        inner_dim = max(center_dim, scatter_dim)

        return max(self._config.size, 2 * (bw + 1) + inner_dim)

    def _get_perimeter_positions(self, grid: np.ndarray, count: int) -> list[tuple[int, int]]:
        """Get evenly spaced positions on perimeter."""
        bw = self._config.border_width
        h, w = grid.shape

        offset = bw + 1
        top, left = offset, offset
        bottom, right = h - offset - 1, w - offset - 1

        # Collect all perimeter positions
        all_positions = []
        for col in range(left, right + 1):
            all_positions.append((top, col))
        for row in range(top + 1, bottom + 1):
            all_positions.append((row, right))
        for col in range(right - 1, left - 1, -1):
            all_positions.append((bottom, col))
        for row in range(bottom - 1, top, -1):
            all_positions.append((row, left))

        # Space them out evenly
        positions = []
        if count > 0 and len(all_positions) > 0:
            step = max(1, len(all_positions) // count)
            for i in range(0, min(count * step, len(all_positions)), step):
                positions.append(all_positions[i])

        return positions[:count]

    def _get_centered_object_positions(self, grid: np.ndarray, count: int) -> list[tuple[int, int]]:
        """Get most centered positions for center objects with 2-cell spacing."""
        import math

        h, w = grid.shape
        center_h = h // 2
        center_w = w // 2

        positions = []

        if count == 1:
            # Single object at exact center
            positions.append((center_h, center_w))
        elif count == 2:
            # Two objects: place them horizontally centered
            positions.append((center_h, center_w - 1))
            positions.append((center_h, center_w + 1))
        elif count == 3:
            # Three objects: horizontal line with center
            positions.append((center_h, center_w - 2))
            positions.append((center_h, center_w))
            positions.append((center_h, center_w + 2))
        elif count == 4:
            # Four objects: 2x2 grid
            positions.append((center_h - 1, center_w - 1))
            positions.append((center_h - 1, center_w + 1))
            positions.append((center_h + 1, center_w - 1))
            positions.append((center_h + 1, center_w + 1))
        else:
            # Larger counts: use grid arrangement
            objects_per_side = math.ceil(math.sqrt(count))
            grid_span = 2 * (objects_per_side - 1)
            start_row = center_h - grid_span // 2
            start_col = center_w - grid_span // 2

            idx = 0
            for i in range(objects_per_side):
                for j in range(objects_per_side):
                    if idx >= count:
                        break
                    row = start_row + i * 2
                    col = start_col + j * 2
                    positions.append((row, col))
                    idx += 1

        return positions

    def _get_occupied_positions(self, grid: np.ndarray) -> list[tuple[int, int]]:
        """Get all non-empty positions in the grid."""
        positions = []
        h, w = grid.shape
        for i in range(h):
            for j in range(w):
                if grid[i, j] != "empty":
                    positions.append((i, j))
        return positions

    def _scatter_objects(self, grid: np.ndarray, objects: list[str], avoid_positions: list[tuple[int, int]]) -> None:
        """Scatter objects randomly with spacing constraints."""
        bw = self._config.border_width
        h, w = grid.shape
        margin = bw + 1

        # Shuffle for random placement order
        shuffled_objects = objects.copy()
        self._rng.shuffle(shuffled_objects)

        placed_positions = avoid_positions.copy()

        for obj in shuffled_objects:
            # Find all valid positions
            valid_positions = []

            for i in range(margin, h - margin):
                for j in range(margin, w - margin):
                    if grid[i, j] != "empty":
                        continue

                    # Check spacing from all occupied positions
                    valid = True
                    for placed_row, placed_col in placed_positions:
                        # Need at least 2 cells distance
                        if abs(i - placed_row) <= 1 and abs(j - placed_col) <= 1:
                            valid = False
                            break

                    if valid:
                        valid_positions.append((i, j))

            if valid_positions:
                # Randomly select from valid positions
                idx = self._rng.choice(len(valid_positions))
                row, col = valid_positions[idx]
                grid[row, col] = obj
                placed_positions.append((row, col))

    def _place_agents(self, grid: np.ndarray, agent_list: list[str]) -> None:
        """Place agents in any empty space."""
        bw = self._config.border_width
        h, w = grid.shape
        margin = bw + 1

        empty_positions = []
        for i in range(margin, h - margin):
            for j in range(margin, w - margin):
                if grid[i, j] == "empty":
                    empty_positions.append((i, j))

        self._rng.shuffle(empty_positions)

        for agent, (row, col) in zip(agent_list, empty_positions):
            grid[row, col] = agent

    def _place_objects(self, grid: np.ndarray, objects: list[str], positions: list[tuple[int, int]]) -> None:
        """Place objects at specified positions."""
        for obj, (row, col) in zip(objects, positions):
            grid[row, col] = obj

    def _get_num_agents(self) -> int:
        """Get total number of agents."""
        if isinstance(self._config.agents, int):
            return self._config.agents
        elif isinstance(self._config.agents, dict):
            return sum(self._config.agents.values())
        return 0

    def _create_object_list(self, objects: dict[str, int]) -> list[str]:
        """Create list from object dict."""
        result = []
        for name, count in objects.items():
            if count > 0:
                result.extend([name] * count)
        return result

    def _create_agent_list(self) -> list[str]:
        """Create list of agent symbols."""
        if isinstance(self._config.agents, int):
            return ["agent.agent"] * self._config.agents
        elif isinstance(self._config.agents, dict):
            return [f"agent.{name}" for name, count in self._config.agents.items() for _ in range(count)]
        return []
