"""
This file defines the VariedTerrainDiverse environment.
It creates a grid world with configurable features including:
  - Large obstacles and small obstacles: randomly generated, connected shapes.
  - Cross obstacles: cross-shaped patterns.
  - Mini labyrinths: maze-like structures (≈11×11) with passages thickened probabilistically,
      and with at least two-cell gaps along the borders; empty cells may be replaced with "heart".
  - Scattered single walls: individual wall cells placed at random empty cells.
  - Blocks: new rectangular objects whose width and height are sampled uniformly between 2 and 14.
      The number of blocks is determined from the style.
  - Altars: the only object placed, whose count is determined by hearts_count.
  - A clumpiness factor that biases object placement.
All objects are placed with at least a one-cell clearance.
If no space is found for a new object, placement is skipped.
The build order is:
    mini labyrinths → obstacles (large, small, crosses) → scattered walls → blocks → altars → agents.
"""

from typing import List, Optional, Tuple

import numpy as np

from metta.common.util.config import Config
from metta.map.scene import Scene


class VariedTerrainParams(Config):
    objects: dict[str, int]
    agents: int = 1
    style: str = "balanced"


class VariedTerrain(Scene[VariedTerrainParams]):
    # Base style parameters for a 60x60 (area=3600) grid.
    # These counts are intentionally moderate.
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
        # New style: maze-like with predominant labyrinth features.
        "maze": {
            "large_obstacles": {"size_range": [10, 25], "count": [0, 2]},  # Disable large obstacles.
            "small_obstacles": {"size_range": [3, 6], "count": [0, 2]},  # Disable small obstacles.
            "crosses": {"count": [0, 2]},  # No cross obstacles.
            "labyrinths": {"count": [10, 20]},  # Increase labyrinth count to generate more maze segments.
            "scattered_walls": {"count": [0, 2]},  # Avoid adding extra walls that could break up maze consistency.
            "blocks": {"count": [0, 2]},  # No rectangular blocks.
            "clumpiness": [0, 2],  # Clumpiness is not necessary when only labyrinths are used.
        },
    }

    def post_init(self):
        style = self.params.style
        if style not in self.STYLE_PARAMETERS:
            raise ValueError(f"Unknown style: '{style}'. Available styles: {list(self.STYLE_PARAMETERS.keys())}")
        base_params = self.STYLE_PARAMETERS[style]
        # Determine scale from the room area relative to a 60x60 (3600 cells) grid.
        area = self.width * self.height
        scale = area / 3600.0

        # Define approximate average cell occupancy for each obstacle type.
        avg_sizes = {
            "large_obstacles": 17.5,  # average of 10 and 25
            "small_obstacles": 4.5,  # average of 3 and 6
            "crosses": 9,  # assumed average area
            "labyrinths": 72,  # rough estimate for labyrinth wall area
            "scattered_walls": 1,
            "blocks": 64,  # approximate average block area (e.g., 8x8)
        }

        # Allowed fraction of the room that obstacles of each type may occupy.
        allowed_fraction = 0.3

        def clamp_count(base_count, avg_size):
            base_count = self.rng.integers(base_count[0], base_count[1])
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
        self._scattered_walls = {
            "count": clamp_count(base_params["scattered_walls"]["count"], avg_sizes["scattered_walls"])
        }
        self._blocks = {"count": clamp_count(base_params["blocks"]["count"], avg_sizes["blocks"])}

    def render(self):
        # Initialize an occupancy mask: False means cell is empty, True means occupied.
        self._occupancy = np.zeros((self.height, self.width), dtype=bool)

        # Place features in order.
        self._place_labyrinths()
        self._place_all_obstacles()
        self._place_scattered_walls()
        self._place_blocks()

        # Place agents.
        for _ in range(self.params.agents):
            pos = self._choose_random_empty()
            if pos is None:
                break
            r, c = pos
            self.grid[r, c] = "agent.agent"
            self._occupancy[r, c] = True

        # Place objects.
        for obj_name, obj_count in self.params.objects.items():
            num_objs_to_place = obj_count - np.where(self.grid == obj_name, 1, 0).sum()
            if num_objs_to_place > 0:
                for _ in range(num_objs_to_place):
                    pos = self._choose_random_empty()
                    if pos is None:
                        break
                    r, c = pos
                    self.grid[r, c] = obj_name
                    self._occupancy[r, c] = True

    def get_labels(self) -> list[str]:
        return [*super().get_labels(), self.params.style]

    # ---------------------------
    # Helper Functions
    # ---------------------------
    def _update_occupancy(self, top_left: Tuple[int, int], pattern: np.ndarray) -> None:
        """
        Updates the occupancy mask for the region where the pattern was placed.
        """
        r, c = top_left
        p_h, p_w = pattern.shape
        self._occupancy[r : r + p_h, c : c + p_w] |= pattern != "empty"

    def _find_candidates(self, region_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Efficiently finds candidate top-left positions where a subregion of shape 'region_shape'
        is completely empty using a sliding window approach on the occupancy mask.
        """
        r_h, r_w = region_shape
        H, W = self._occupancy.shape
        if H < r_h or W < r_w:
            return []
        # Create a view of all submatrices of shape (r_h, r_w)
        shape = (H - r_h + 1, W - r_w + 1, r_h, r_w)
        strides = self._occupancy.strides * 2
        submats = np.lib.stride_tricks.as_strided(self._occupancy, shape=shape, strides=strides)
        # Sum over each submatrix; candidate if sum == 0 (i.e., completely empty)
        window_sums = submats.sum(axis=(2, 3))
        candidates = np.argwhere(window_sums == 0)
        return [tuple(idx) for idx in candidates]

    def _choose_random_empty(self) -> Optional[Tuple[int, int]]:
        """
        Efficiently returns a random empty position using the occupancy mask.
        """
        empty_flat = np.flatnonzero(~self._occupancy)
        if empty_flat.size == 0:
            return None
        idx = self.rng.integers(0, empty_flat.size)
        return np.unravel_index(empty_flat[idx], self._occupancy.shape)

    def _place_candidate_region(self, pattern: np.ndarray, clearance: int = 0) -> bool:
        """
        Attempts to place the given pattern in a candidate region that is completely empty,
        taking into account a clearance border around the pattern.
        """
        p_h, p_w = pattern.shape
        eff_h, eff_w = p_h + 2 * clearance, p_w + 2 * clearance
        candidates = self._find_candidates((eff_h, eff_w))
        if candidates:
            r, c = candidates[self.rng.integers(0, len(candidates))]
            # Place pattern with clearance offset.
            self.grid[r + clearance : r + clearance + p_h, c + clearance : c + clearance + p_w] = pattern
            self._update_occupancy((r + clearance, c + clearance), pattern)
            return True
        return False

    # ---------------------------
    # Placement Routines
    # ---------------------------
    def _place_labyrinths(self):
        labyrinth_count = self._labyrinths.get("count", 0)
        for _ in range(labyrinth_count):
            pattern = self._generate_labyrinth_pattern()
            assert len(pattern.shape) == 2
            candidates = self._find_candidates(pattern.shape)
            if candidates:
                r, c = candidates[self.rng.integers(0, len(candidates))]
                self.grid[r : r + pattern.shape[0], c : c + pattern.shape[1]] = pattern
                self._update_occupancy((r, c), pattern)

    def _place_all_obstacles(self):
        clearance = 1
        # Place large obstacles.
        large_count = self._large_obstacles.get("count", 0)
        low_large, high_large = self._large_obstacles.get("size_range", [10, 25])
        for _ in range(large_count):
            target = self.rng.integers(low_large, high_large + 1)
            pattern = self._generate_random_shape(target)
            self._place_candidate_region(pattern, clearance)
        # Place small obstacles.
        small_count = self._small_obstacles.get("count", 0)
        low_small, high_small = self._small_obstacles.get("size_range", [3, 6])
        for _ in range(small_count):
            target = self.rng.integers(low_small, high_small + 1)
            pattern = self._generate_random_shape(target)
            self._place_candidate_region(pattern, clearance)
        # Place cross obstacles (with no extra clearance).
        crosses_count = self._crosses.get("count", 0)
        for _ in range(crosses_count):
            pattern = self._generate_cross_pattern()
            assert len(pattern.shape) == 2
            candidates = self._find_candidates(pattern.shape)
            if candidates:
                r, c = candidates[self.rng.integers(0, len(candidates))]
                self.grid[r : r + pattern.shape[0], c : c + pattern.shape[1]] = pattern
                self._update_occupancy((r, c), pattern)

    def _place_scattered_walls(self):
        count = self._scattered_walls.get("count", 0)
        empty_flat = np.flatnonzero(~self._occupancy)
        num_to_place = min(count, empty_flat.size)
        if num_to_place == 0:
            return
        chosen_flat = self.rng.choice(empty_flat, size=num_to_place, replace=False)
        r_coords, c_coords = np.unravel_index(chosen_flat, self.grid.shape)
        self.grid[r_coords, c_coords] = "wall"
        self._occupancy[r_coords, c_coords] = True

    def _place_blocks(self):
        """
        Places rectangular block objects on the grid.
        For each block, the width and height are sampled uniformly between 2 and 14.
        The number of blocks is determined by self._blocks["count"].
        The block is placed in a candidate region that is completely empty.
        """
        block_count = self._blocks.get("count", 0)
        for _ in range(block_count):
            block_w = self.rng.integers(2, 15)  # 2 to 14 inclusive.
            block_h = self.rng.integers(2, 15)
            candidates = self._find_candidates((block_h, block_w))
            if candidates:
                r, c = candidates[self.rng.integers(0, len(candidates))]
                self.grid[r : r + block_h, c : c + block_w] = "wall"
                block_pattern = np.full((block_h, block_w), "wall", dtype=object)
                self._update_occupancy((r, c), block_pattern)

    # ---------------------------
    # Pattern Generation Functions
    # ---------------------------
    def _generate_random_shape(self, num_blocks: int) -> np.ndarray:
        shape_cells = {(0, 0)}
        while len(shape_cells) < num_blocks:
            candidates = []
            for r, c in shape_cells:
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    candidate = (r + dr, c + dc)
                    if candidate not in shape_cells:
                        candidates.append(candidate)
            if not candidates:
                break
            new_cell = candidates[self.rng.integers(0, len(candidates))]
            shape_cells.add(new_cell)
        min_r = min(r for r, _ in shape_cells)
        min_c = min(c for _, c in shape_cells)
        max_r = max(r for r, _ in shape_cells)
        max_c = max(c for _, c in shape_cells)
        pattern = np.full((max_r - min_r + 1, max_c - min_c + 1), "empty", dtype=object)
        for r, c in shape_cells:
            pattern[r - min_r, c - min_c] = "wall"
        return pattern

    def _generate_cross_pattern(self) -> np.ndarray:
        cross_w = self.rng.integers(1, 9)
        cross_h = self.rng.integers(1, 9)
        pattern = np.full((cross_h, cross_w), "empty", dtype=object)
        center_row = cross_h // 2
        center_col = cross_w // 2
        pattern[center_row, :] = "wall"
        pattern[:, center_col] = "wall"
        return pattern

    def _generate_labyrinth_pattern(self) -> np.ndarray:
        # Choose dimensions between 11 and 13, then clamp to 11 and force odd.
        h = int(self.rng.integers(11, 26))
        w = int(self.rng.integers(11, 26))
        if h % 2 == 0:
            h -= 1
        if w % 2 == 0:
            w -= 1

        maze = np.full((h, w), "wall", dtype=object)
        start = (1, 1)
        maze[start] = "empty"
        stack = [start]
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        while stack:
            r, c = stack[-1]
            neighbors = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and maze[nr, nc] == "wall":
                    neighbors.append((nr, nc))
            if neighbors:
                next_cell = neighbors[self.rng.integers(0, len(neighbors))]
                nr, nc = next_cell
                wall_r, wall_c = r + (nr - r) // 2, c + (nc - c) // 2
                maze[wall_r, wall_c] = "empty"
                maze[nr, nc] = "empty"
                stack.append(next_cell)
            else:
                stack.pop()

        # Ensure each border has at least two contiguous empty cells.
        if w > 3 and not self._has_gap(maze[0, 1 : w - 1]):
            maze[0, 1:3] = "empty"
        if w > 3 and not self._has_gap(maze[h - 1, 1 : w - 1]):
            maze[h - 1, 1:3] = "empty"
        if h > 3 and not self._has_gap(maze[1 : h - 1, 0]):
            maze[1:3, 0] = "empty"
        if h > 3 and not self._has_gap(maze[1 : h - 1, w - 1]):
            maze[1:3, w - 1] = "empty"

        # Scatter hearts in empty cells with 1% probability.
        for i in range(h):
            for j in range(w):
                if maze[i, j] == "empty" and self.rng.random() < 0.03:
                    maze[i, j] = "altar"

            # Apply thickening based on a random probability between 0.3 and 1.0.
        thick_prob = 0.7 * self.rng.random()
        maze_thick = maze.copy()
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if maze[i, j] == "empty":
                    if self.rng.random() < thick_prob and j + 1 < w:
                        maze_thick[i, j + 1] = "empty"
                    if self.rng.random() < thick_prob and i + 1 < h:
                        maze_thick[i + 1, j] = "empty"
        maze = maze_thick
        return maze

    def _has_gap(self, line: np.ndarray) -> bool:
        contiguous = 0
        for cell in line:
            contiguous = contiguous + 1 if cell == "empty" else 0
            if contiguous >= 2:
                return True
        return False
