"""
Streamlined VariedTerrain optimization with dynamic parameter generation.
Key optimizations:
- Fast maze generation using boolean arrays
- Vectorized operations
- Dynamic style parameters
- Efficient candidate finding
"""

from typing import List, Optional, Tuple
import time
import numpy as np
from omegaconf import DictConfig
from mettagrid.room.room import Room


class VariedTerrain(Room):
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
        self.style = style
        self._objects = objects

        # Generate parameters dynamically based on style and area
        area = width * height
        scale = area / 3600.0  # Base on 60x60 grid
        self._setup_parameters(scale)

    def _setup_parameters(self, scale):
        """Generate all parameters dynamically based on style."""
        if self.style == "all-sparse":
            multiplier = 0.3
        elif self.style == "balanced":
            multiplier = 1.0
        elif self.style == "dense":
            multiplier = 2.0
        elif self.style == "maze":
            multiplier = 1.0
        else:
            multiplier = 1.0

        # Scale counts based on area and style
        base_scale = max(1, int(scale * multiplier))

        if self.style == "maze":
            # Maze style: mostly labyrinths
            self._large_count = 0
            self._small_count = 0
            self._cross_count = 0
            self._labyrinth_count = self._rng.integers(10, 20) * base_scale
            self._scattered_count = 0
            self._block_count = 0
        elif self.style == "dense":
            # Dense style: lots of everything
            self._large_count = self._rng.integers(8, 15) * base_scale
            self._small_count = self._rng.integers(8, 15) * base_scale
            self._cross_count = self._rng.integers(7, 15) * base_scale
            self._labyrinth_count = self._rng.integers(6, 15) * base_scale
            self._scattered_count = self._rng.integers(40, 60) * base_scale
            self._block_count = self._rng.integers(5, 15) * base_scale
        elif self.style == "all-sparse":
            # Sparse style: minimal everything
            self._large_count = self._rng.integers(0, 2)
            self._small_count = self._rng.integers(0, 2)
            self._cross_count = self._rng.integers(0, 2)
            self._labyrinth_count = self._rng.integers(0, 2)
            self._scattered_count = self._rng.integers(0, 2)
            self._block_count = self._rng.integers(0, 2)
        else:  # balanced
            self._large_count = self._rng.integers(3, 7) * base_scale
            self._small_count = self._rng.integers(3, 7) * base_scale
            self._cross_count = self._rng.integers(3, 7) * base_scale
            self._labyrinth_count = self._rng.integers(3, 7) * base_scale
            self._scattered_count = self._rng.integers(3, 7) * base_scale
            self._block_count = self._rng.integers(3, 7) * base_scale

    def _build(self) -> np.ndarray:
        start = time.time()

        # Create grid and occupancy tracking
        grid = np.full((self._height, self._width), "empty", dtype=object)
        self._occupancy = np.zeros((self._height, self._width), dtype=bool)

        # Place features
        self._place_labyrinths(grid)
        self._place_obstacles(grid)
        self._place_scattered_walls(grid)
        self._place_blocks(grid)
        self._place_agents_and_objects(grid)

        end = time.time()
        print(f"Time taken to build varied terrain {self.style}: {end - start:.3f} seconds")
        return grid

    def _place_labyrinths(self, grid):
        """Fast labyrinth placement."""
        for _ in range(self._labyrinth_count):
            pattern = self._generate_fast_maze()
            if self._place_pattern(grid, pattern):
                continue

    def _place_obstacles(self, grid):
        """Place all obstacle types."""
        # Large obstacles
        for _ in range(self._large_count):
            size = self._rng.integers(10, 26)
            pattern = self._generate_shape(size)
            self._place_pattern(grid, pattern, clearance=1)

        # Small obstacles
        for _ in range(self._small_count):
            size = self._rng.integers(3, 7)
            pattern = self._generate_shape(size)
            self._place_pattern(grid, pattern, clearance=1)

        # Crosses
        for _ in range(self._cross_count):
            pattern = self._generate_cross()
            self._place_pattern(grid, pattern)

    def _place_scattered_walls(self, grid):
        """Vectorized scattered wall placement."""
        if self._scattered_count == 0:
            return

        empty_positions = np.argwhere(~self._occupancy)
        if len(empty_positions) == 0:
            return

        count = min(self._scattered_count, len(empty_positions))
        chosen = self._rng.choice(len(empty_positions), size=count, replace=False)
        positions = empty_positions[chosen]

        grid[positions[:, 0], positions[:, 1]] = "wall"
        self._occupancy[positions[:, 0], positions[:, 1]] = True

    def _place_blocks(self, grid):
        """Place rectangular blocks."""
        for _ in range(self._block_count):
            h, w = self._rng.integers(2, 15, size=2)
            if self._place_rectangle(grid, h, w):
                continue

    def _place_agents_and_objects(self, grid):
        """Place agents and objects efficiently."""
        # Prepare agents
        if self._teams is None:
            agents = ["agent.agent"] * (self._agents if isinstance(self._agents, int) else 1)
        else:
            agents = []
            per_team = self._agents // len(self._teams) if isinstance(self._agents, int) else 1
            for team in self._teams:
                agents.extend([f"agent.{team}"] * per_team)

        # Get empty positions
        empty_positions = np.argwhere(~self._occupancy)
        self._rng.shuffle(empty_positions)
        idx = 0

        # Place agents
        for agent in agents:
            if idx >= len(empty_positions):
                break
            r, c = empty_positions[idx]
            grid[r, c] = agent
            self._occupancy[r, c] = True
            idx += 1

        # Place objects
        for obj_name, obj_count in self._objects.items():
            existing = np.sum(grid == obj_name)
            needed = obj_count - existing

            for _ in range(needed):
                if idx >= len(empty_positions):
                    break
                r, c = empty_positions[idx]
                grid[r, c] = obj_name
                self._occupancy[r, c] = True
                idx += 1

    def _generate_fast_maze(self):
        """Ultra-fast maze generation using boolean arrays."""
        # Random odd dimensions
        h = self._rng.integers(11, 21)
        w = self._rng.integers(11, 21)
        h = h if h % 2 == 1 else h - 1
        w = w if w % 2 == 1 else w - 1

        # Boolean maze: True=wall, False=empty
        maze = np.ones((h, w), dtype=bool)

        # DFS maze generation
        start = (1, 1)
        maze[start] = False
        stack = [start]
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

        while stack:
            r, c = stack[-1]
            neighbors = []

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 < nr < h-1 and 0 < nc < w-1 and maze[nr, nc]:
                    neighbors.append((nr, nc))

            if neighbors:
                nr, nc = neighbors[self._rng.integers(0, len(neighbors))]
                maze[(r + nr) // 2, (c + nc) // 2] = False  # Carve wall
                maze[nr, nc] = False
                stack.append((nr, nc))
            else:
                stack.pop()

        # Ensure border gaps
        if w > 3:
            maze[0, 1:3] = False
            maze[h-1, 1:3] = False
        if h > 3:
            maze[1:3, 0] = False
            maze[1:3, w-1] = False

        # Convert to object array and add altars
        result = np.where(maze, "wall", "empty").astype(object)
        empty_mask = ~maze
        altar_positions = empty_mask & (self._rng.random((h, w)) < 0.03)
        result[altar_positions] = "altar"

        return result

    def _generate_shape(self, target_size):
        """Generate connected random shape."""
        shape_cells = {(0, 0)}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while len(shape_cells) < target_size:
            candidates = []
            for r, c in shape_cells:
                for dr, dc in directions:
                    new_pos = (r + dr, c + dc)
                    if new_pos not in shape_cells:
                        candidates.append(new_pos)

            if not candidates:
                break

            shape_cells.add(candidates[self._rng.integers(0, len(candidates))])

        # Create bounding box
        coords = np.array(list(shape_cells))
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)

        h, w = max_coords - min_coords + 1
        pattern = np.full((h, w), "empty", dtype=object)

        for r, c in shape_cells:
            pattern[r - min_coords[0], c - min_coords[1]] = "wall"

        return pattern

    def _generate_cross(self):
        """Generate cross pattern."""
        h, w = self._rng.integers(3, 9, size=2)
        pattern = np.full((h, w), "empty", dtype=object)
        pattern[h//2, :] = "wall"  # Horizontal line
        pattern[:, w//2] = "wall"  # Vertical line
        return pattern

    def _place_pattern(self, grid, pattern, clearance=0):
        """Place pattern with optional clearance."""
        p_h, p_w = pattern.shape
        req_h, req_w = p_h + 2*clearance, p_w + 2*clearance

        # Find valid positions
        candidates = []
        for r in range(self._height - req_h + 1):
            for c in range(self._width - req_w + 1):
                if not self._occupancy[r:r+req_h, c:c+req_w].any():
                    candidates.append((r, c))

        if not candidates:
            return False

        # Place pattern
        r, c = candidates[self._rng.integers(0, len(candidates))]
        pr, pc = r + clearance, c + clearance
        grid[pr:pr+p_h, pc:pc+p_w] = pattern

        # Update occupancy
        mask = pattern != "empty"
        self._occupancy[pr:pr+p_h, pc:pc+p_w] |= mask
        return True

    def _place_rectangle(self, grid, h, w):
        """Place rectangular block."""
        candidates = []
        for r in range(self._height - h + 1):
            for c in range(self._width - w + 1):
                if not self._occupancy[r:r+h, c:c+w].any():
                    candidates.append((r, c))

        if not candidates:
            return False

        r, c = candidates[self._rng.integers(0, len(candidates))]
        grid[r:r+h, c:c+w] = "wall"
        self._occupancy[r:r+h, c:c+w] = True
        return True


# End of streamlined VariedTerrain class
