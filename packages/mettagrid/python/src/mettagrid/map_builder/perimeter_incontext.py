import random
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
        Densities: can be None, "sparse", "balanced", or "dense"

        Given the width and height, the number of obstacles and obstacle size is determined by the density.
        """

        seed: Optional[int] = None

        width: int = 7
        height: int = 7
        objects: dict[str, int] = {}
        density: str = "no-terrain"
        agents: int | dict[str, int] = 1
        border_width: int = 0
        border_object: str = "wall"

        chain_length: int = 2
        num_sinks: int = 0
        dir: Optional[str] = None

    def __init__(self, config: Config):
        self._config = config
        self._rng = np.random.default_rng(self._config.seed)
        # Pre-compute obstacle shapes to avoid recreating them
        self._obstacle_shapes_cache = {}

    def _create_obstacle_shapes(self, obstacle_type: str, size: int = 2):
        """Create obstacle shape patterns with caching."""
        cache_key = (obstacle_type, size)
        if cache_key not in self._obstacle_shapes_cache:
            if obstacle_type == "square":
                shape = self._create_square_obstacle(size)
            elif obstacle_type == "cross":
                shape = self._create_cross_obstacle(size)
            elif obstacle_type == "L":
                shape = self._create_l_obstacle(size)
            else:
                # Default to single block
                shape = np.array([["wall"]])
            self._obstacle_shapes_cache[cache_key] = shape

        return self._obstacle_shapes_cache[cache_key]

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
        elif density == "dense":
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

    def _can_reach_perimeter_optimized(self, grid: np.ndarray, start_i: int, start_j: int) -> bool:
        """Optimized BFS pathfinding with early termination and vectorized operations."""
        if grid[start_i, start_j] == "wall":
            return False

        height, width = grid.shape

        # Use a single array for visited tracking
        visited = np.zeros(height * width, dtype=bool)
        queue = deque([(start_i, start_j)])
        visited[start_i * width + start_j] = True

        # Pre-compute perimeter check
        def is_perimeter(i, j):
            return i == 0 or i == height - 1 or j == 0 or j == width - 1

        while queue:
            i, j = queue.popleft()

            # Check if we've reached the perimeter
            if is_perimeter(i, j):
                return True

            # Check all four directions with bounds checking
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    flat_idx = ni * width + nj
                    if not visited[flat_idx] and grid[ni, nj] != "wall":
                        visited[flat_idx] = True
                        queue.append((ni, nj))

        return False

    def _get_valid_positions_vectorized(self, grid_shape, obstacle_shape, avoid_mask):
        """Vectorized computation of valid obstacle positions."""
        grid_h, grid_w = grid_shape
        shape_h, shape_w = obstacle_shape.shape

        valid_positions = []

        # Use numpy broadcasting for efficient overlap checking
        for i in range(grid_h - shape_h + 1):
            for j in range(grid_w - shape_w + 1):
                # Check if this position overlaps with avoid_mask
                region = avoid_mask[i : i + shape_h, j : j + shape_w]
                if not np.any(region):
                    valid_positions.append((i, j))

        return valid_positions

    def _place_obstacle_optimized(self, grid: np.ndarray, obstacle_shape: np.ndarray, avoid_mask: np.ndarray) -> bool:
        """Optimized obstacle placement with reduced redundant operations."""
        # Get valid positions once
        valid_positions = self._get_valid_positions_vectorized(grid.shape, obstacle_shape, avoid_mask)

        if not valid_positions:
            return False

        center_i, center_j = grid.shape[0] // 2, grid.shape[1] // 2

        # Shuffle positions once
        self._rng.shuffle(valid_positions)

        # Pre-extract wall positions from obstacle shape
        wall_positions = np.where(obstacle_shape == "wall")
        wall_coords = list(zip(wall_positions[0], wall_positions[1], strict=True))

        for i, j in valid_positions:
            # Create temporary grid more efficiently
            temp_grid = grid.copy()

            # Apply obstacle using pre-computed wall coordinates
            for di, dj in wall_coords:
                temp_grid[i + di, j + dj] = "wall"

            # Check if agent can still reach perimeter
            if self._can_reach_perimeter_optimized(temp_grid, center_i, center_j):
                # Place the obstacle permanently using the same coordinates
                for di, dj in wall_coords:
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

        # Create perimeter mask more efficiently using vectorized operations
        perimeter_mask = np.zeros((height, width), dtype=bool)

        # Use array slicing for efficiency
        perimeter_mask[0, :] = True  # Top row
        perimeter_mask[-1, :] = True  # Bottom row
        perimeter_mask[:, 0] = True  # Left column
        perimeter_mask[:, -1] = True  # Right column

        # Exclude corners if grid is large enough
        if height >= 2 and width >= 2:
            corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
            for i, j in corners:
                perimeter_mask[i, j] = False

        # Find empty perimeter cells for objects using vectorized operations
        empty_perimeter_mask = (grid == "empty") & perimeter_mask
        empty_perimeter_indices = np.flatnonzero(empty_perimeter_mask.ravel())

        # Prepare and place objects on perimeter
        object_symbols = []
        for obj_name, count in self._config.objects.items():
            object_symbols.extend([obj_name] * count)

        if object_symbols and len(empty_perimeter_indices) > 0:
            object_symbols = np.array(object_symbols, dtype=str)
            self._rng.shuffle(object_symbols)
            self._rng.shuffle(empty_perimeter_indices)

            num_placeable = min(len(object_symbols), len(empty_perimeter_indices))
            if num_placeable > 0:
                flat_grid = grid.ravel()
                selected_indices = empty_perimeter_indices[:num_placeable]
                flat_grid[selected_indices] = object_symbols[:num_placeable]
                grid = flat_grid.reshape(height, width)

        # Place obstacles if specified with optimizations
        density = self._config.density
        if density == "no-terrain":
            density = None
        obstacle_type = random.choice(["square", "cross", "L"])

        if obstacle_type and density:
            densities_to_try = [density, "balanced", "sparse"]
            obstacles_placed = 0

            for fallback_density in densities_to_try:
                num_obstacles, obstacle_size = self._get_density_config(fallback_density, inner_area, obstacle_type)

                # Create avoid mask more efficiently
                avoid_mask = perimeter_mask.copy()  # Start with perimeter mask

                # Add inner perimeter efficiently
                if height > 2 and width > 2:
                    inner_perimeter = np.zeros((height, width), dtype=bool)
                    inner_perimeter[1, :] = True
                    inner_perimeter[-2, :] = True
                    inner_perimeter[:, 1] = True
                    inner_perimeter[:, -2] = True

                    # Exclude inner corners
                    inner_corners = [(1, 1), (1, -2), (-2, 1), (-2, -2)]
                    for i, j in inner_corners:
                        inner_perimeter[i, j] = False

                    avoid_mask |= inner_perimeter

                # Reserve center for agent
                center_i, center_j = height // 2, width // 2
                avoid_mask[center_i, center_j] = True

                # Try to place obstacles with optimized method
                obstacle_shape = self._create_obstacle_shapes(obstacle_type, obstacle_size)

                for _ in range(num_obstacles):
                    success = self._place_obstacle_optimized(grid, obstacle_shape, avoid_mask)
                    if success:
                        obstacles_placed += 1
                    else:
                        break

                if obstacles_placed > 0:
                    break

        # Place agent in center efficiently
        center_i, center_j = height // 2, width // 2
        grid[center_i, center_j] = agents[0]

        return GameMap(grid)
