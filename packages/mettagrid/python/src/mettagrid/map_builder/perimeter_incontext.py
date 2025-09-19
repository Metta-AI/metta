from collections import deque
from typing import Optional

import numpy as np

from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from mettagrid.map_builder.utils import draw_border


class PerimeterInContextMapBuilder(MapBuilder):
    class Config(MapBuilderConfig["PerimeterInContextMapBuilder"]):
        """
        Configuration for building a mini in-context learning map.

        Objects appear on the perimeter, and the agent appears in the center.

        Always a single agent in this map.

        Obstacle types: can be None, "square", "cross", or "L"
        Densities: can be None, "sparse", "balanced", or "high"

        Given the width and height, the number of obstacles and obstacle size is determined by the density.
        """

        seed: Optional[int] = None

        width: int = 7
        height: int = 7
        objects: dict[str, int] = {}
        obstacle_type: Optional[str] = None
        density: Optional[str] = None
        agents: int | dict[str, int] = 1
        border_width: int = 0
        border_object: str = "wall"

    def __init__(self, config: Config):
        self._config = config
        self._rng = np.random.default_rng(self._config.seed)

    def _create_obstacle_shapes(self, obstacle_type: str, size: int = 2):
        """Create obstacle shape patterns."""
        if obstacle_type == "square":
            return self._create_square_obstacle(size)
        elif obstacle_type == "cross":
            return self._create_cross_obstacle(size)
        elif obstacle_type == "L":
            return self._create_l_obstacle(size)
        else:
            # Default to single block
            return np.array([["wall"]])

    def _create_square_obstacle(self, size: int):
        """Create a square obstacle of given size."""
        return np.full((size, size), "wall", dtype="<U50")

    def _create_cross_obstacle(self, size: int):
        """Create a cross-shaped obstacle."""
        cross = np.full((size * 2 - 1, size * 2 - 1), "empty", dtype="<U50")
        # Horizontal line
        cross[size - 1, :] = "wall"
        # Vertical line
        cross[:, size - 1] = "wall"
        return cross

    def _create_l_obstacle(self, size: int):
        """Create an L-shaped obstacle."""
        l_shape = np.full((size, size), "empty", dtype="<U50")
        # Vertical part
        l_shape[:, 0] = "wall"
        # Horizontal part
        l_shape[size - 1, :] = "wall"
        return l_shape

    def _get_density_config(self, density: str, inner_area: int, obstacle_type: str):
        """Get obstacle configuration based on density and obstacle type."""
        if density == "sparse":
            num_obstacles = max(1, inner_area // 30)  # Very few obstacles
            # Ensure cross and L shapes have minimum size for recognizable shape
            if obstacle_type in ["cross", "L"]:
                obstacle_size = 2  # Minimum size for recognizable cross/L
            else:
                obstacle_size = 1
        elif density == "balanced":
            num_obstacles = max(2, inner_area // 12)  # Moderate obstacles
            obstacle_size = 2
        elif density == "high":
            # Adjust for obstacle type - larger shapes need fewer obstacles
            if obstacle_type == "cross":
                num_obstacles = max(2, inner_area // 15)  # Crosses are large (3x3)
            elif obstacle_type == "L":
                num_obstacles = max(2, inner_area // 12)  # L-shapes are medium (2x2)
            else:  # square
                num_obstacles = max(3, inner_area // 8)  # Squares are small (2x2)
            obstacle_size = 2
        else:
            num_obstacles = 0
            obstacle_size = 1

        return num_obstacles, obstacle_size

    def _can_reach_perimeter(self, grid: np.ndarray, start_i: int, start_j: int) -> bool:
        """Check if there's a path from start position to perimeter using BFS."""
        if grid[start_i, start_j] == "wall":
            return False

        height, width = grid.shape
        visited = np.zeros((height, width), dtype=bool)
        queue = deque([(start_i, start_j)])
        visited[start_i, start_j] = True

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            i, j = queue.popleft()

            # Check if we've reached the perimeter
            if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                return True

            # Check all directions
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width and not visited[ni, nj] and grid[ni, nj] != "wall":
                    visited[ni, nj] = True
                    queue.append((ni, nj))

        return False

    def _place_obstacle(self, grid: np.ndarray, obstacle_shape: np.ndarray, avoid_mask: np.ndarray) -> bool:
        """Try to place an obstacle shape on the grid, avoiding specified areas."""
        shape_h, shape_w = obstacle_shape.shape
        grid_h, grid_w = grid.shape

        # Find valid placement positions
        valid_positions = []
        for i in range(grid_h - shape_h + 1):
            for j in range(grid_w - shape_w + 1):
                # Check if this position is valid (no overlap with avoid_mask)
                if not np.any(avoid_mask[i : i + shape_h, j : j + shape_w]):
                    valid_positions.append((i, j))

        if not valid_positions:
            return False

        # Try placing the obstacle and check if agent can still escape
        center_i, center_j = grid_h // 2, grid_w // 2

        # Shuffle positions to try them randomly
        self._rng.shuffle(valid_positions)

        for i, j in valid_positions:
            # Temporarily place the obstacle
            temp_grid = grid.copy()
            for di in range(shape_h):
                for dj in range(shape_w):
                    if obstacle_shape[di, dj] == "wall":
                        temp_grid[i + di, j + dj] = "wall"

            # Check if agent can still reach perimeter
            if self._can_reach_perimeter(temp_grid, center_i, center_j):
                # Place the obstacle permanently
                for di in range(shape_h):
                    for dj in range(shape_w):
                        if obstacle_shape[di, dj] == "wall":
                            grid[i + di, j + dj] = "wall"
                return True

        return False

    def build(self):
        height = self._config.height
        width = self._config.width

        # Create empty grid
        grid = np.full((height, width), "empty", dtype="<U50")

        # Draw border first if needed
        if self._config.border_width > 0:
            draw_border(grid, self._config.border_width, self._config.border_object)

        # Calculate inner area where objects can be placed
        if self._config.border_width > 0:
            inner_height = max(0, self._config.height - 2 * self._config.border_width)
            inner_width = max(0, self._config.width - 2 * self._config.border_width)
            inner_area = inner_height * inner_width
        else:
            inner_height = self._config.height
            inner_width = self._config.width
            inner_area = self._config.width * self._config.height

        if inner_area <= 0:
            return GameMap(grid)  # No room for objects, return border-only grid

        # always a single agent
        agents = ["agent.agent"]

        # Find perimeter cells (cells touching the border)
        perimeter_mask = np.zeros((height, width), dtype=bool)

        # Top and bottom rows
        perimeter_mask[0, :] = True
        perimeter_mask[height - 1, :] = True

        # Left and right columns
        perimeter_mask[:, 0] = True
        perimeter_mask[:, width - 1] = True

        # Exclude the four corners from placement
        if height >= 2 and width >= 2:
            perimeter_mask[0, 0] = False
            perimeter_mask[0, width - 1] = False
            perimeter_mask[height - 1, 0] = False
            perimeter_mask[height - 1, width - 1] = False

        # Find empty perimeter cells for objects
        empty_perimeter_mask = (grid == "empty") & perimeter_mask
        empty_perimeter_indices = np.where(empty_perimeter_mask.flatten())[0]

        # Prepare objects for perimeter placement
        object_symbols = []
        for obj_name, count in self._config.objects.items():
            object_symbols.extend([obj_name] * count)

        flat_grid = grid.flatten()

        # Place objects on perimeter
        object_symbols = np.array(object_symbols).astype(str)
        self._rng.shuffle(object_symbols)
        self._rng.shuffle(empty_perimeter_indices)
        selected_perimeter_indices = empty_perimeter_indices[: len(object_symbols)]
        flat_grid[selected_perimeter_indices] = object_symbols

        grid = flat_grid.reshape(height, width)

        # Place obstacles if specified (only for PerimeterInContextMapBuilderWithObstacles)
        obstacle_type = getattr(self._config, "obstacle_type", None)
        density = getattr(self._config, "density", None)

        if obstacle_type and density:
            # Try different densities if the requested one fails
            densities_to_try = [density, "balanced", "sparse"]
            obstacles_placed = 0

            for fallback_density in densities_to_try:
                num_obstacles, obstacle_size = self._get_density_config(fallback_density, inner_area, obstacle_type)

                # Create mask of areas to avoid
                avoid_mask = np.zeros((height, width), dtype=bool)

                # Avoid perimeter (where converters are placed)
                avoid_mask |= perimeter_mask

                # Avoid inner perimeter (cells just inside the border) to ensure accessibility
                inner_perimeter_mask = np.zeros((height, width), dtype=bool)
                if height > 2 and width > 2:
                    # Inner perimeter: one cell in from the border
                    inner_perimeter_mask[1, :] = True  # Top inner row
                    inner_perimeter_mask[height - 2, :] = True  # Bottom inner row
                    inner_perimeter_mask[:, 1] = True  # Left inner column
                    inner_perimeter_mask[:, width - 2] = True  # Right inner column

                    # Exclude corners from inner perimeter
                    inner_perimeter_mask[1, 1] = False
                    inner_perimeter_mask[1, width - 2] = False
                    inner_perimeter_mask[height - 2, 1] = False
                    inner_perimeter_mask[height - 2, width - 2] = False

                avoid_mask |= inner_perimeter_mask

                # Reserve center for agent
                center_i, center_j = height // 2, width // 2
                avoid_mask[center_i, center_j] = True

                # Try to place obstacles
                obstacles_placed = 0
                for _ in range(num_obstacles):
                    obstacle_shape = self._create_obstacle_shapes(obstacle_type, obstacle_size)
                    success = self._place_obstacle(grid, obstacle_shape, avoid_mask)
                    if success:
                        obstacles_placed += 1
                    else:
                        break  # No more valid positions

                # If we placed at least one obstacle, we're done
                if obstacles_placed > 0:
                    break

        # place agent in center
        center_index = (height // 2) * width + (width // 2)
        flat_grid = grid.flatten()
        flat_grid[center_index] = agents[0]

        grid = flat_grid.reshape(height, width)

        return GameMap(grid)
