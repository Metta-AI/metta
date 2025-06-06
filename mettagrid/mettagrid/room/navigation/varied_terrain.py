import time
import numpy as np
from typing import List, Optional, Tuple
from omegaconf import DictConfig
from mettagrid.room.room import Room


class VariedTerrain(Room):
    # Base style parameters for a 60×60 (area=3600) grid
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

        # Pre-compute scaled parameters once
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

        # Prepare agent list efficiently
        agents = self._prepare_agents()

        # Initialize grid and occupancy mask
        grid = np.full((self._height, self._width), "empty", dtype=object)
        self._occupancy = np.zeros((self._height, self._width), dtype=bool)

        # Place all feature types
        grid = self._place_labyrinths(grid)
        grid = self._place_all_obstacles(grid)
        grid = self._place_scattered_walls(grid)
        grid = self._place_blocks(grid)

        # Place agents and objects last
        self._place_agents_and_objects(grid, agents)

        end = time.time()
        print(f"Time taken to build varied terrain '{self.style}': {end - start:.3f} seconds")
        return grid

    def _prepare_agents(self) -> List[str]:
        """Efficiently prepare agent list."""
        if self._teams is None:
            if isinstance(self._agents, int):
                return ["agent.agent"] * self._agents
        else:
            agents = []
            teams_count = len(self._teams)
            if isinstance(self._agents, int) and teams_count > 0:
                per_team = self._agents // teams_count
                for team in self._teams:
                    agents.extend(["agent." + team] * per_team)
                return agents
        return []

    def _place_agents_and_objects(self, grid: np.ndarray, agents: List[str]) -> None:
        """Place agents and objects efficiently using batched empty-position sampling."""
        empty_positions = self._get_all_empty_positions()
        placement_idx = 0

        # Place agents first
        for agent in agents:
            if placement_idx >= len(empty_positions):
                break
            r, c = empty_positions[placement_idx]
            grid[r, c] = agent
            self._occupancy[r, c] = True
            placement_idx += 1

        # Place objects based on desired counts
        for obj_name, target_count in self._objects.items():
            existing = np.sum(grid == obj_name)
            to_place = max(0, target_count - existing)
            for _ in range(to_place):
                if placement_idx >= len(empty_positions):
                    break
                r, c = empty_positions[placement_idx]
                grid[r, c] = obj_name
                self._occupancy[r, c] = True
                placement_idx += 1

    def _get_all_empty_positions(self) -> List[Tuple[int, int]]:
        """Get all empty positions in random order (vectorized)."""
        empty_flat = np.flatnonzero(~self._occupancy)
        self._rng.shuffle(empty_flat)
        return [np.unravel_index(idx, self._occupancy.shape) for idx in empty_flat]

    # ---------------------------
    # Optimized Helper Methods
    # ---------------------------

    def _update_occupancy_vectorized(self, top_left: Tuple[int, int], pattern: np.ndarray) -> None:
        """Update occupancy map for a placed pattern in one shot."""
        r0, c0 = top_left
        ph, pw = pattern.shape
        mask = (pattern != "empty")
        self._occupancy[r0 : r0 + ph, c0 : c0 + pw] |= mask

    def _find_candidates_optimized(self, region_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find all top-left placements where region_shape fits entirely on empty cells."""
        from scipy import ndimage

        rh, rw = region_shape
        H, W = self._occupancy.shape

        if H < rh or W < rw:
            return []

        # Convolve occupancy with an all-ones kernel of size region_shape
        kernel = np.ones((rh, rw), dtype=int)
        conv_result = ndimage.convolve(self._occupancy.astype(int), kernel, mode="constant", cval=0)

        # Valid placements are those where the convolution is zero
        valid = np.argwhere(conv_result[: H - rh + 1, : W - rw + 1] == 0)
        return [(int(r), int(c)) for r, c in valid]

    def _place_candidate_region_optimized(
        self, grid: np.ndarray, pattern: np.ndarray, clearance: int = 0
    ) -> bool:
        """Attempt to place a rectangular pattern with clearance around it."""
        ph, pw = pattern.shape
        eff_h, eff_w = ph + 2 * clearance, pw + 2 * clearance

        try:
            candidates = self._find_candidates_optimized((eff_h, eff_w))
        except ImportError:
            candidates = self._find_candidates((eff_h, eff_w))

        if not candidates:
            return False

        r, c = candidates[self._rng.integers(0, len(candidates))]
        pr, pc = r + clearance, c + clearance
        grid[pr : pr + ph, pc : pc + pw] = pattern
        self._update_occupancy_vectorized((pr, pc), pattern)
        return True

    def _find_candidates(self, region_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Fallback candidate finder using numpy as_strided or brute force."""
        rh, rw = region_shape
        H, W = self._occupancy.shape

        if H < rh or W < rw:
            return []

        try:
            shape = (H - rh + 1, W - rw + 1, rh, rw)
            strides = self._occupancy.strides * 2
            submats = np.lib.stride_tricks.as_strided(self._occupancy, shape=shape, strides=strides)
            window_sums = submats.sum(axis=(2, 3))
            coords = np.argwhere(window_sums == 0)
            return [(int(r), int(c)) for r, c in coords]
        except Exception:
            candidates = []
            for r in range(H - rh + 1):
                for c in range(W - rw + 1):
                    if not self._occupancy[r : r + rh, c : c + rw].any():
                        candidates.append((r, c))
            return candidates

    # ---------------------------
    # Optimized Placement Routines
    # ---------------------------

    def _place_labyrinths(self, grid: np.ndarray) -> np.ndarray:
        """Place a number of random mazes (labyrinths)."""
        count = self._labyrinths.get("count", 0)
        for _ in range(count):
            pattern = self._generate_labyrinth_pattern_optimized()
            candidates = self._find_candidates(pattern.shape)
            if not candidates:
                continue
            r, c = candidates[self._rng.integers(0, len(candidates))]
            grid[r : r + pattern.shape[0], c : c + pattern.shape[1]] = pattern
            self._update_occupancy_vectorized((r, c), pattern)
        return grid

    def _place_all_obstacles(self, grid: np.ndarray) -> np.ndarray:
        """Place large, small obstacles and crosses."""
        clearance = 1
        self._place_obstacles_batch(grid, self._large_obstacles, clearance)
        self._place_obstacles_batch(grid, self._small_obstacles, clearance)
        self._place_crosses_batch(grid)
        return grid

    def _place_obstacles_batch(self, grid: np.ndarray, config: dict, clearance: int):
        """Batch placement of rectangular or random-shaped obstacles."""
        count = config.get("count", 0)
        size_range = config.get("size_range", [3, 6])
        for _ in range(count):
            size = self._rng.integers(size_range[0], size_range[1] + 1)
            pattern = self._generate_random_shape_optimized(size)
            try:
                self._place_candidate_region_optimized(grid, pattern, clearance)
            except ImportError:
                self._place_candidate_region(grid, pattern, clearance)

    def _place_crosses_batch(self, grid: np.ndarray):
        """Place a number of crosses."""
        count = self._crosses.get("count", 0)
        for _ in range(count):
            pattern = self._generate_cross_pattern()
            candidates = self._find_candidates(pattern.shape)
            if not candidates:
                continue
            r, c = candidates[self._rng.integers(0, len(candidates))]
            grid[r : r + pattern.shape[0], c : c + pattern.shape[1]] = pattern
            self._update_occupancy_vectorized((r, c), pattern)

    def _place_scattered_walls(self, grid: np.ndarray) -> np.ndarray:
        """Vectorized placement of single-cell walls scattered around."""
        count = self._scattered_walls.get("count", 0)
        empty_flat = np.flatnonzero(~self._occupancy)
        if empty_flat.size == 0:
            return grid
        to_place = min(count, empty_flat.size)
        chosen = self._rng.choice(empty_flat, size=to_place, replace=False)
        r_coords, c_coords = np.unravel_index(chosen, grid.shape)
        grid[r_coords, c_coords] = "wall"
        self._occupancy[r_coords, c_coords] = True
        return grid

    def _place_blocks(self, grid: np.ndarray) -> np.ndarray:
        """Place rectangular blocks of walls."""
        count = self._blocks.get("count", 0)
        for _ in range(count):
            bw = self._rng.integers(2, 15)
            bh = self._rng.integers(2, 15)
            candidates = self._find_candidates((bh, bw))
            if not candidates:
                continue
            r, c = candidates[self._rng.integers(0, len(candidates))]
            grid[r : r + bh, c : c + bw] = "wall"
            self._occupancy[r : r + bh, c : c + bw] = True
        return grid

    # ---------------------------
    # Optimized Pattern Generators
    # ---------------------------

    def _generate_random_shape_optimized(self, num_blocks: int) -> np.ndarray:
        """
        Generate a connected random shape of size num_blocks using a flood-fill approach,
        then return its minimal bounding-box pattern.
        """
        shape_cells = {(0, 0)}
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while len(shape_cells) < num_blocks:
            candidates = set()
            for (r, c) in shape_cells:
                for dr, dc in directions:
                    nbr = (r + dr, c + dc)
                    if nbr not in shape_cells:
                        candidates.add(nbr)
            if not candidates:
                break

            candidates_list = list(candidates)
            new_cell = candidates_list[self._rng.integers(0, len(candidates_list))]
            shape_cells.add(new_cell)

        coords = np.array(list(shape_cells))
        min_rc = coords.min(axis=0)
        max_rc = coords.max(axis=0)
        ph, pw = (max_rc - min_rc + 1)
        pattern = np.full((ph, pw), "empty", dtype=object)

        for (r, c) in shape_cells:
            pattern[r - min_rc[0], c - min_rc[1]] = "wall"

        return pattern

    def _generate_cross_pattern(self) -> np.ndarray:
        """Generate a simple cross (vertical + horizontal line) of random odd dimensions."""
        ch = self._rng.integers(1, 9)
        cw = self._rng.integers(1, 9)
        pattern = np.full((ch, cw), "empty", dtype=object)
        center_r = ch // 2
        center_c = cw // 2
        pattern[center_r, :] = "wall"
        pattern[:, center_c] = "wall"
        return pattern

    def _generate_labyrinth_pattern_optimized(self) -> np.ndarray:
        """
        Generate a random maze using iterative DFS and carve passages.
        Returns a 2D array of "wall"/"empty" with occasional "altar" cells.
        """
        # Choose odd dimensions in [11, 25]
        h = self._rng.integers(11, 26)
        w = self._rng.integers(11, 26)
        h -= (h % 2 == 0)
        w -= (w % 2 == 0)

        maze = np.ones((h, w), dtype=bool)  # True=wall, False=empty
        stack = [(1, 1)]
        maze[1, 1] = False

        directions = np.array([(-2, 0), (2, 0), (0, -2), (0, 2)])

        while stack:
            r, c = stack[-1]
            # Compute potential neighbors two cells away
            nbrs = np.array([r, c]) + directions
            valid = (
                (nbrs[:, 0] >= 1)
                & (nbrs[:, 0] < h - 1)
                & (nbrs[:, 1] >= 1)
                & (nbrs[:, 1] < w - 1)
            )
            nbrs = nbrs[valid]
            if nbrs.size == 0:
                stack.pop()
                continue

            # Filter only truly unvisited walls
            is_wall = maze[nbrs[:, 0], nbrs[:, 1]]
            unvisited = nbrs[is_wall]
            if unvisited.size == 0:
                stack.pop()
                continue

            idx = self._rng.integers(0, len(unvisited))
            nr, nc = tuple(unvisited[idx])
            # Carve the wall between current and next
            wr, wc = (r + nr) // 2, (c + nc) // 2
            maze[wr, wc] = False
            maze[nr, nc] = False
            stack.append((nr, nc))

        # Ensure border has at least one gap on each side
        self._ensure_border_gaps_vectorized(maze)

        # Convert boolean maze to object array
        result = np.where(maze, "wall", "empty").astype(object)

        # Randomly sprinkle "altar" (hearts) on a small fraction of empty cells
        empty_mask = ~maze
        heart_prob = 0.03
        rand_mat = self._rng.random((h, w))
        altar_positions = (empty_mask) & (rand_mat < heart_prob)
        result[altar_positions] = "altar"

        # Apply thickening to carve additional random passages
        result = self._apply_maze_thickening_optimized(result)

        return result

    def _ensure_border_gaps_vectorized(self, maze: np.ndarray) -> None:
        """Guarantee at least one two-cell gap on each outer border if possible."""
        h, w = maze.shape
        # Top & bottom rows
        if w > 3:
            for br in (0, h - 1):
                border_line = ~maze[br, 1 : w - 1]
                if not self._has_gap_vectorized(border_line):
                    maze[br, 1 : 3] = False

        # Left & right columns
        if h > 3:
            for bc in (0, w - 1):
                border_line = ~maze[1 : h - 1, bc]
                if not self._has_gap_vectorized(border_line):
                    maze[1 : 3, bc] = False

    def _has_gap_vectorized(self, line: np.ndarray) -> bool:
        """Detect two consecutive True entries (empty) using convolution."""
        if len(line) < 2:
            return False
        kernel = np.ones(2, dtype=int)
        conv_res = np.convolve(line.astype(int), kernel, mode="valid")
        return np.any(conv_res == 2)

    def _apply_maze_thickening_optimized(self, grid: np.ndarray) -> np.ndarray:
        """
        Randomly carve additional openings in the maze:
        For each interior empty cell, possibly carve toward right or down if wall exists.
        """
        h, w = grid.shape
        # Choose a random thickening probability
        thick_prob = 0.7 * self._rng.random()
        empty_mask = (grid == "empty")

        interior_empty = empty_mask[1 : h - 1, 1 : w - 1]
        right_wall = grid[1 : h - 1, 2 : w] != "empty"
        down_wall = grid[2 : h, 1 : w - 1] != "empty"

        rand_matrix = self._rng.random((h - 2, w - 2))
        thicken_right = (rand_matrix < thick_prob) & interior_empty & right_wall
        thicken_down = (rand_matrix < thick_prob) & interior_empty & down_wall

        grid[1 : h - 1, 2 : w][thicken_right] = "empty"
        grid[2 : h, 1 : w - 1][thicken_down] = "empty"

        return grid

    # ---------------------------
    # Fallback Methods (unchanged)
    # ---------------------------

    def _place_candidate_region(self, grid: np.ndarray, pattern: np.ndarray, clearance: int = 0) -> bool:
        """Fallback placement if scipy is unavailable."""
        ph, pw = pattern.shape
        eff_h, eff_w = ph + 2 * clearance, pw + 2 * clearance
        candidates = self._find_candidates((eff_h, eff_w))
        if not candidates:
            return False
        r, c = candidates[self._rng.integers(0, len(candidates))]
        pr, pc = r + clearance, c + clearance
        grid[pr : pr + ph, pc : pc + pw] = pattern
        self._update_occupancy_vectorized((pr, pc), pattern)
        return True

    def _has_gap(self, line: np.ndarray) -> bool:
        """Original gap detection (fallback)."""
        contiguous = 0
        for cell in line:
            contiguous = contiguous + 1 if cell else 0
            if contiguous >= 2:
                return True
        return False
