"""
Optimized VariedTerrainDiverse environment with significant performance improvements.
Key optimizations:
- Vectorized operations for maze generation
- Pre-computed candidate lists
- Efficient numpy operations
- Reduced redundant calculations
- Optimized occupancy tracking
"""

import time
import numpy as np
from typing import List, Optional, Tuple, Set
from omegaconf import DictConfig
from mettagrid.room.room import Room


class VariedTerrain(Room):
    # Base style parameters for a 60x60 (area=3600) grid.
    STYLE_PARAMETERS = {
        "all-sparse": {
            "large_obstacles": {"size_range": [10, 25], "count": [0, 2]},
            "small_obstacles": {"size_range": [3, 6], "count": [0, 2]},
            "crosses": {"count": [0, 2]},
            "labyrinths": {"count": [0, 2]},
            "scattered_walls": {"count": [0, 2]},
            "blocks": {"count": [0, 2]},
            "clumpiness": [0, 2],
        },
        "balanced": {
            "large_obstacles": {"size_range": [10, 25], "count": [3, 7]},
            "small_obstacles": {"size_range": [3, 6], "count": [3, 7]},
            "crosses": {"count": [3, 7]},
            "labyrinths": {"count": [3, 7]},
            "scattered_walls": {"count": [3, 7]},
            "blocks": {"count": [3, 7]},
            "clumpiness": [1, 3],
        },
        "dense": {
            "large_obstacles": {"size_range": [10, 25], "count": [8, 15]},
            "small_obstacles": {"size_range": [3, 6], "count": [8, 15]},
            "crosses": {"count": [7, 15]},
            "labyrinths": {"count": [6, 15]},
            "scattered_walls": {"count": [40, 60]},
            "blocks": {"count": [5, 15]},
            "clumpiness": [2, 6],
        },
        "maze": {
            "large_obstacles": {"size_range": [10, 25], "count": [0, 2]},
            "small_obstacles": {"size_range": [3, 6], "count": [0, 2]},
            "crosses": {"count": [0, 2]},
            "labyrinths": {"count": [10, 20]},
            "scattered_walls": {"count": [0, 2]},
            "blocks": {"count": [0, 2]},
            "clumpiness": [0, 2],
        },
    }

    def __init__(
        self,
        width: int,
        height: int,
        objects: DictConfig,
        agents: int | dict = 1,
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: str = "wall",
        occupancy_threshold: float = 0.66,
        style: str = "balanced",
        teams: list | None = None,
    ):
        super().__init__(border_width=border_width, border_object=border_object, labels=[style])
        self.set_size_labels(width, height)
        self._rng = np.random.default_rng(seed)
        self._width = width
        self._height = height
        self._agents = agents
        self._teams = teams
        self._occupancy_threshold = occupancy_threshold

        if style not in self.STYLE_PARAMETERS:
            raise ValueError(f"Unknown style: '{style}'. Available styles: {list(self.STYLE_PARAMETERS.keys())}")

        self.style = style
        base_params = self.STYLE_PARAMETERS[style]
        area = width * height
        scale = area / 3600.0

        # Pre-compute scaled parameters
        self._setup_scaled_parameters(base_params, scale, area)
        self._objects = objects

    def _setup_scaled_parameters(self, base_params, scale, area):
        """Pre-compute all scaled parameters to avoid repeated calculations."""
        avg_sizes = {
            "large_obstacles": 17.5,
            "small_obstacles": 4.5,
            "crosses": 9,
            "labyrinths": 72,
            "scattered_walls": 1,
            "blocks": 64,
        }
        allowed_fraction = 0.3

        def clamp_count(base_count, avg_size):
            base_count = self._rng.integers(base_count[0], base_count[1])
            scaled = int(base_count * scale)
            max_allowed = int((allowed_fraction * area) / avg_size)
            return min(scaled, max_allowed) if scaled > 0 else 0

        self._large_obstacles = {
            "size_range": base_params["large_obstacles"]["size_range"],
            "count": clamp_count(base_params["large_obstacles"]["count"], avg_sizes["large_obstacles"]),
        }
        self._small_obstacles = {
            "size_range": base_params["small_obstacles"]["size_range"],
            "count": clamp_count(base_params["small_obstacles"]["count"], avg_sizes["small_obstacles"]),
        }
        self._crosses = {"count": clamp_count(base_params["crosses"]["count"], avg_sizes["crosses"])}
        self._labyrinths = {"count": clamp_count(base_params["labyrinths"]["count"], avg_sizes["labyrinths"])}
        self._scattered_walls = {"count": clamp_count(base_params["scattered_walls"]["count"], avg_sizes["scattered_walls"])}
        self._blocks = {"count": clamp_count(base_params["blocks"]["count"], avg_sizes["blocks"])}

    def _build(self) -> np.ndarray:
        start = time.time()

        # Prepare agent symbols efficiently
        agents = self._prepare_agents()

        # Create empty grid and occupancy mask
        grid = np.full((self._height, self._width), "empty", dtype=object)
        self._occupancy = np.zeros((self._height, self._width), dtype=bool)

        # Pre-compute empty positions for faster lookups
        self._empty_positions_cache = None

        # Place features in order
        grid = self._place_labyrinths(grid)
        grid = self._place_all_obstacles(grid)
        grid = self._place_scattered_walls(grid)
        grid = self._place_blocks(grid)

        # Place agents and objects efficiently
        self._place_agents_and_objects(grid, agents)

        end = time.time()
        print(f"Time taken to build varied terrain {self.style}: {end - start:.3f} seconds")
        return grid

    def _prepare_agents(self):
        """Efficiently prepare agent list."""
        if self._teams is None:
            if isinstance(self._agents, int):
                return ["agent.agent"] * self._agents
        else:
            agents = []
            agents_per_team = self._agents // len(self._teams)
            for team in self._teams:
                agents.extend(["agent." + team] * agents_per_team)
            return agents
        return []

    def _place_agents_and_objects(self, grid, agents):
        """Place agents and objects efficiently using batch operations."""
        # Get all empty positions at once
        empty_positions = self._get_all_empty_positions()
        placement_idx = 0

        # Place agents
        for agent in agents:
            if placement_idx >= len(empty_positions):
                break
            r, c = empty_positions[placement_idx]
            grid[r, c] = agent
            self._occupancy[r, c] = True
            placement_idx += 1

        # Place objects
        for obj_name, obj_count in self._objects.items():
            existing_count = np.sum(grid == obj_name)
            num_to_place = obj_count - existing_count

            for _ in range(num_to_place):
                if placement_idx >= len(empty_positions):
                    break
                r, c = empty_positions[placement_idx]
                grid[r, c] = obj_name
                self._occupancy[r, c] = True
                placement_idx += 1

    def _get_all_empty_positions(self):
        """Get all empty positions and shuffle them once for random placement."""
        empty_flat = np.flatnonzero(~self._occupancy)
        self._rng.shuffle(empty_flat)
        return [np.unravel_index(idx, self._occupancy.shape) for idx in empty_flat]

    # ---------------------------
    # Optimized Helper Functions
    # ---------------------------

    def _update_occupancy_vectorized(self, top_left: Tuple[int, int], pattern: np.ndarray) -> None:
        """Vectorized occupancy update."""
        r, c = top_left
        p_h, p_w = pattern.shape
        mask = pattern != "empty"
        self._occupancy[r:r + p_h, c:c + p_w] |= mask

    def _find_candidates_optimized(self, region_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Optimized candidate finding with early termination."""
        r_h, r_w = region_shape
        H, W = self._occupancy.shape

        if H < r_h or W < r_w:
            return []

        # Use convolution for faster sliding window
        from scipy import ndimage
        kernel = np.ones((r_h, r_w))
        conv_result = ndimage.convolve(self._occupancy.astype(int), kernel, mode='constant')

        # Find positions where convolution result is 0 (completely empty)
        valid_positions = np.argwhere(conv_result[:H-r_h+1, :W-r_w+1] == 0)
        return [tuple(pos) for pos in valid_positions]

    def _place_candidate_region_optimized(self, grid: np.ndarray, pattern: np.ndarray, clearance: int = 0) -> bool:
        """Optimized region placement with fewer operations."""
        p_h, p_w = pattern.shape
        eff_h, eff_w = p_h + 2 * clearance, p_w + 2 * clearance

        try:
            candidates = self._find_candidates_optimized((eff_h, eff_w))
        except ImportError:
            # Fallback to original method if scipy not available
            candidates = self._find_candidates((eff_h, eff_w))

        if candidates:
            r, c = candidates[self._rng.integers(0, len(candidates))]
            # Place pattern with clearance offset
            pr, pc = r + clearance, c + clearance
            grid[pr:pr + p_h, pc:pc + p_w] = pattern
            self._update_occupancy_vectorized((pr, pc), pattern)
            return True
        return False

    def _find_candidates(self, region_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Original candidate finding method (fallback)."""
        r_h, r_w = region_shape
        H, W = self._occupancy.shape
        if H < r_h or W < r_w:
            return []

        shape = (H - r_h + 1, W - r_w + 1, r_h, r_w)
        strides = self._occupancy.strides * 2
        try:
            submats = np.lib.stride_tricks.as_strided(self._occupancy, shape=shape, strides=strides)
            window_sums = submats.sum(axis=(2, 3))
            candidates = np.argwhere(window_sums == 0)
            return [tuple(idx) for idx in candidates]
        except ValueError:
            # Fallback for edge cases
            candidates = []
            for r in range(H - r_h + 1):
                for c in range(W - r_w + 1):
                    if not self._occupancy[r:r+r_h, c:c+r_w].any():
                        candidates.append((r, c))
            return candidates

    # ---------------------------
    # Optimized Placement Routines
    # ---------------------------

    def _place_labyrinths(self, grid: np.ndarray) -> np.ndarray:
        """Optimized labyrinth placement."""
        labyrinth_count = self._labyrinths.get("count", 0)

        for _ in range(labyrinth_count):
            pattern = self._generate_labyrinth_pattern_optimized()
            candidates = self._find_candidates(pattern.shape)

            if candidates:
                r, c = candidates[self._rng.integers(0, len(candidates))]
                grid[r:r + pattern.shape[0], c:c + pattern.shape[1]] = pattern
                self._update_occupancy_vectorized((r, c), pattern)

        return grid

    def _place_all_obstacles(self, grid: np.ndarray) -> np.ndarray:
        """Optimized obstacle placement."""
        clearance = 1

        # Place large obstacles
        self._place_obstacles_batch(grid, self._large_obstacles, clearance)
        # Place small obstacles
        self._place_obstacles_batch(grid, self._small_obstacles, clearance)
        # Place crosses
        self._place_crosses_batch(grid)

        return grid

    def _place_obstacles_batch(self, grid: np.ndarray, obstacle_config: dict, clearance: int):
        """Batch placement of obstacles."""
        count = obstacle_config.get("count", 0)
        size_range = obstacle_config.get("size_range", [3, 6])

        for _ in range(count):
            target = self._rng.integers(size_range[0], size_range[1] + 1)
            pattern = self._generate_random_shape_optimized(target)
            try:
                self._place_candidate_region_optimized(grid, pattern, clearance)
            except ImportError:
                self._place_candidate_region(grid, pattern, clearance)

    def _place_crosses_batch(self, grid: np.ndarray):
        """Optimized cross placement."""
        crosses_count = self._crosses.get("count", 0)

        for _ in range(crosses_count):
            pattern = self._generate_cross_pattern()
            candidates = self._find_candidates(pattern.shape)

            if candidates:
                r, c = candidates[self._rng.integers(0, len(candidates))]
                grid[r:r + pattern.shape[0], c:c + pattern.shape[1]] = pattern
                self._update_occupancy_vectorized((r, c), pattern)

    def _place_scattered_walls(self, grid: np.ndarray) -> np.ndarray:
        """Vectorized scattered wall placement."""
        count = self._scattered_walls.get("count", 0)
        empty_flat = np.flatnonzero(~self._occupancy)
        num_to_place = min(count, empty_flat.size)

        if num_to_place == 0:
            return grid

        chosen_flat = self._rng.choice(empty_flat, size=num_to_place, replace=False)
        r_coords, c_coords = np.unravel_index(chosen_flat, grid.shape)

        # Vectorized assignment
        grid[r_coords, c_coords] = "wall"
        self._occupancy[r_coords, c_coords] = True

        return grid

    def _place_blocks(self, grid: np.ndarray) -> np.ndarray:
        """Optimized block placement."""
        block_count = self._blocks.get("count", 0)

        for _ in range(block_count):
            block_w = self._rng.integers(2, 15)
            block_h = self._rng.integers(2, 15)
            candidates = self._find_candidates((block_h, block_w))

            if candidates:
                r, c = candidates[self._rng.integers(0, len(candidates))]
                # Vectorized block placement
                grid[r:r + block_h, c:c + block_w] = "wall"
                self._occupancy[r:r + block_h, c:c + block_w] = True

        return grid

    # ---------------------------
    # Optimized Pattern Generation
    # ---------------------------

    def _generate_random_shape_optimized(self, num_blocks: int) -> np.ndarray:
        """Optimized random shape generation using sets and vectorized operations."""
        shape_cells = {(0, 0)}
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while len(shape_cells) < num_blocks:
            # Pre-compute all candidates at once
            candidates = set()
            for r, c in shape_cells:
                for dr, dc in directions:
                    candidate = (r + dr, c + dc)
                    if candidate not in shape_cells:
                        candidates.add(candidate)

            if not candidates:
                break

            # Convert to list for random selection
            candidates_list = list(candidates)
            new_cell = candidates_list[self._rng.integers(0, len(candidates_list))]
            shape_cells.add(new_cell)

        # Vectorized bounding box calculation
        coords = np.array(list(shape_cells))
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)

        pattern_shape = tuple(max_coords - min_coords + 1)
        pattern = np.full(pattern_shape, "empty", dtype=object)

        # Vectorized pattern filling
        for r, c in shape_cells:
            pattern[r - min_coords[0], c - min_coords[1]] = "wall"

        return pattern

    def _generate_cross_pattern(self) -> np.ndarray:
        """Generate cross pattern (no changes needed - already efficient)."""
        cross_w = self._rng.integers(1, 9)
        cross_h = self._rng.integers(1, 9)
        pattern = np.full((cross_h, cross_w), "empty", dtype=object)
        center_row = cross_h // 2
        center_col = cross_w // 2
        pattern[center_row, :] = "wall"
        pattern[:, center_col] = "wall"
        return pattern

    def _generate_labyrinth_pattern_optimized(self) -> np.ndarray:
        """Highly optimized maze generation using vectorized operations and efficient algorithms."""
        # Random dimensions, ensure odd
        h = self._rng.integers(11, 26)
        w = self._rng.integers(11, 26)
        h = h if h % 2 == 1 else h - 1
        w = w if w % 2 == 1 else w - 1

        # Initialize maze - all walls
        maze = np.full((h, w), True, dtype=bool)  # True = wall, False = empty

        # Randomized DFS with optimized neighbor checking
        start = (1, 1)
        maze[start] = False
        stack = [start]
        directions = np.array([(-2, 0), (2, 0), (0, -2), (0, 2)])

        while stack:
            r, c = stack[-1]

            # Vectorized neighbor finding
            next_coords = np.array([r, c]) + directions
            valid_mask = (
                (next_coords[:, 0] >= 0) & (next_coords[:, 0] < h) &
                (next_coords[:, 1] >= 0) & (next_coords[:, 1] < w)
            )
            valid_coords = next_coords[valid_mask]

            if len(valid_coords) > 0:
                # Check which neighbors are walls
                wall_mask = maze[valid_coords[:, 0], valid_coords[:, 1]]
                unvisited = valid_coords[wall_mask]

                if len(unvisited) > 0:
                    # Choose random unvisited neighbor
                    next_idx = self._rng.integers(0, len(unvisited))
                    nr, nc = unvisited[next_idx]

                    # Carve path
                    wall_r, wall_c = (r + nr) // 2, (c + nc) // 2
                    maze[wall_r, wall_c] = False
                    maze[nr, nc] = False
                    stack.append((nr, nc))
                else:
                    stack.pop()
            else:
                stack.pop()

        # Ensure border gaps efficiently
        self._ensure_border_gaps_vectorized(maze, h, w)

        # Convert boolean maze to object array and add hearts
        result_maze = np.where(maze, "wall", "empty").astype(object)

        # Vectorized heart placement
        empty_mask = ~maze
        heart_prob = 0.03
        heart_positions = empty_mask & (self._rng.random((h, w)) < heart_prob)
        result_maze[heart_positions] = "altar"

        # Optimized thickening
        result_maze = self._apply_maze_thickening_optimized(result_maze, h, w)

        return result_maze

    def _ensure_border_gaps_vectorized(self, maze: np.ndarray, h: int, w: int):
        """Efficiently ensure border gaps using vectorized operations."""
        # Top and bottom borders
        if w > 3:
            for border_row in [0, h-1]:
                border = ~maze[border_row, 1:w-1]  # empty cells
                if not self._has_gap_vectorized(border):
                    maze[border_row, 1:3] = False

        # Left and right borders
        if h > 3:
            for border_col in [0, w-1]:
                border = ~maze[1:h-1, border_col]  # empty cells
                if not self._has_gap_vectorized(border):
                    maze[1:3, border_col] = False

    def _has_gap_vectorized(self, line: np.ndarray) -> bool:
        """Vectorized gap detection."""
        if len(line) < 2:
            return False

        # Use convolution to find consecutive True values
        kernel = np.ones(2)
        conv_result = np.convolve(line.astype(int), kernel, mode='valid')
        return np.any(conv_result == 2)  # Two consecutive empty cells

    def _apply_maze_thickening_optimized(self, maze: np.ndarray, h: int, w: int) -> np.ndarray:
        """Optimized maze thickening using vectorized operations."""
        thick_prob = 0.7 * self._rng.random()

        # Create masks for positions to potentially thicken
        empty_mask = (maze == "empty")
        interior_mask = empty_mask[1:h-1, 1:w-1]

        # Random thickening decisions
        thicken_right = self._rng.random((h-2, w-2)) < thick_prob
        thicken_down = self._rng.random((h-2, w-2)) < thick_prob

        # Apply thickening where valid
        valid_right = interior_mask & (maze[1:h-1, 2:w] != "empty")
        valid_down = interior_mask & (maze[2:h, 1:w-1] != "empty")

        thicken_right_final = thicken_right & valid_right
        thicken_down_final = thicken_down & valid_down

        # Apply thickening
        maze[1:h-1, 2:w][thicken_right_final] = "empty"
        maze[2:h, 1:w-1][thicken_down_final] = "empty"

        return maze

    # Keep original methods for fallback compatibility
    def _place_candidate_region(self, grid: np.ndarray, pattern: np.ndarray, clearance: int = 0) -> bool:
        """Original method for fallback."""
        p_h, p_w = pattern.shape
        eff_h, eff_w = p_h + 2 * clearance, p_w + 2 * clearance
        candidates = self._find_candidates((eff_h, eff_w))
        if candidates:
            r, c = candidates[self._rng.integers(0, len(candidates))]
            grid[r + clearance:r + clearance + p_h, c + clearance:c + clearance + p_w] = pattern
            self._update_occupancy_vectorized((r + clearance, c + clearance), pattern)
            return True
        return False

    def _has_gap(self, line: np.ndarray) -> bool:
        """Original gap detection method."""
        contiguous = 0
        for cell in line:
            contiguous = contiguous + 1 if cell == "empty" else 0
            if contiguous >= 2:
                return True
        return False


# End of optimized VariedTerrain class implementation
