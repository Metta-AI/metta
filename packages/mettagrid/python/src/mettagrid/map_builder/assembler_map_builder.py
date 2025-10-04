from typing import Optional

import numpy as np

from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from mettagrid.map_builder.utils import draw_border


class AssemblerMapBuilder(MapBuilder):
    class Config(MapBuilderConfig["AssemblerMapBuilder"]):
        seed: Optional[int] = None
        width: int = 10
        height: int = 10
        objects: dict[str, int] = {}
        agents: int | dict[str, int] = 0
        border_width: int = 0
        border_object: str = "wall"
        terrain: str = "no-terrain"  # "", "sparse", "balanced", "dense"

    def __init__(self, config: Config):
        self._config = config
        self._rng = np.random.default_rng(self._config.seed)
        self._shape_cache: dict[tuple[str, int], np.ndarray] = {}

    def build(self) -> GameMap:
        """Build a complete game map with terrain, objects, and agents."""
        if self._config.seed is not None:
            self._rng = np.random.default_rng(self._config.seed)

        grid = self._create_base_grid()
        self._add_terrain(grid)
        self._add_objects(grid)
        self._add_agents(grid)

        return GameMap(grid)

    # ========== Grid Setup ==========

    def _create_base_grid(self) -> np.ndarray:
        """Create empty grid with border."""
        grid = np.full((self._config.height, self._config.width), "empty", dtype="<U50")

        if self._config.border_width > 0:
            draw_border(grid, self._config.border_width, self._config.border_object)

        return grid

    # ========== Terrain Placement ==========

    def _add_terrain(self, grid: np.ndarray) -> None:
        """Add terrain obstacles to the grid."""
        inner_bounds = self._get_inner_bounds(grid)
        if inner_bounds is None:
            return

        num_obstacles = self._calculate_num_obstacles(inner_bounds)

        for _ in range(num_obstacles):
            self._try_place_obstacle(grid, inner_bounds)

    def _calculate_num_obstacles(self, bounds: tuple[int, int, int, int]) -> int:
        """Calculate number of obstacles based on terrain density setting."""
        top, bottom, left, right = bounds
        inner_area = (bottom - top + 1) * (right - left + 1)

        terrain = self._config.terrain or "no-terrain"
        density_map = {
            "sparse": inner_area // 40,
            "balanced": inner_area // 22,
            "dense": inner_area // 14,
        }

        return max(density_map.get(terrain, 0), 0 if terrain == "no-terrain" else 1)

    def _try_place_obstacle(self, grid: np.ndarray, bounds: tuple[int, int, int, int], max_tries: int = 200) -> bool:
        """Attempt to place a single obstacle on the grid."""
        top, bottom, left, right = bounds
        shape = self._choose_random_obstacle()
        sh, sw = shape.shape

        i_min, i_max = top, bottom - sh + 1
        j_min, j_max = left, right - sw + 1

        if i_max < i_min or j_max < j_min:
            return False

        for _ in range(max_tries):
            i = self._rng.integers(i_min, i_max + 1)
            j = self._rng.integers(j_min, j_max + 1)

            region = grid[i : i + sh, j : j + sw]

            if np.all(region == "empty"):
                wall_mask = shape == "wall"
                if wall_mask.any():
                    region[wall_mask] = "wall"
                    return True

        return False

    def _choose_random_obstacle(self) -> np.ndarray:
        """Randomly select an obstacle shape."""
        kinds = ["block", "square", "L", "cross"]
        probs = [0.40, 0.30, 0.20, 0.10]
        kind = self._rng.choice(kinds, p=probs)
        return self._get_shape(kind, size=2)

    # ========== Object Placement ==========

    def _add_objects(self, grid: np.ndarray) -> None:
        """Place all objects on the grid with empty boundaries around each."""
        object_list = self._create_object_list()
        if not object_list:
            return

        valid_positions = self._collect_valid_positions(grid)
        if len(valid_positions) == 0:
            return

        self._rng.shuffle(valid_positions)

        object_mask = np.zeros(grid.shape, dtype=bool)

        for obj_symbol in object_list:
            placed = False
            for row, col in valid_positions:
                if self._can_place_object(grid, object_mask, row, col):
                    grid[row, col] = obj_symbol
                    object_mask[row - 1 : row + 2, col - 1 : col + 2] = True
                    placed = True
                    break

            if not placed:
                break

    def _collect_valid_positions(self, grid: np.ndarray) -> list[tuple[int, int]]:
        """Collect all positions where objects could potentially be placed."""
        bounds = self._get_object_bounds(grid)
        if bounds is None:
            return []

        top, bottom, left, right = bounds
        walls = grid == "wall"
        forbidden = self._dilate_bool(walls, radius=1)

        valid_positions = []
        for i in range(top, bottom + 1):
            for j in range(left, right + 1):
                neighborhood = slice(i - 1, i + 2), slice(j - 1, j + 2)
                grid_neighborhood = grid[neighborhood]
                forbidden_neighborhood = forbidden[neighborhood]

                if np.all(grid_neighborhood == "empty") and not np.any(forbidden_neighborhood):
                    valid_positions.append((i, j))

        return valid_positions

    def _can_place_object(self, grid: np.ndarray, object_mask: np.ndarray, i: int, j: int) -> bool:
        """Check if an object can be placed at position (i, j)."""
        neighborhood = slice(i - 1, i + 2), slice(j - 1, j + 2)
        return grid[i, j] == "empty" and np.all(grid[neighborhood] == "empty") and not np.any(object_mask[neighborhood])

    def _create_object_list(self) -> list[str]:
        """Flatten object dictionary into a list of object symbols."""
        return [name for name, count in self._config.objects.items() if count > 0 for _ in range(count)]

    # ========== Agent Placement ==========

    def _add_agents(self, grid: np.ndarray) -> None:
        """Place agents on empty cells."""
        agent_list = self._create_agent_list()
        if not agent_list:
            return

        empty_positions = np.argwhere(grid == "empty")
        if len(empty_positions) == 0:
            return

        self._rng.shuffle(empty_positions)

        num_to_place = min(len(agent_list), len(empty_positions))
        for i in range(num_to_place):
            row, col = empty_positions[i]
            grid[row, col] = agent_list[i]

    def _create_agent_list(self) -> list[str]:
        """Create list of agent symbols from config."""
        if isinstance(self._config.agents, int):
            return ["agent.agent"] * self._config.agents
        elif isinstance(self._config.agents, dict):
            return [f"agent.{name}" for name, count in self._config.agents.items() for _ in range(count)]
        else:
            raise ValueError(f"Invalid agents configuration: {self._config.agents}")

    # ========== Shape Generation ==========

    def _get_shape(self, kind: str, size: int) -> np.ndarray:
        """Get or create a shape with caching."""
        key = (kind, size)
        if key not in self._shape_cache:
            self._shape_cache[key] = self._create_shape(kind, size)
        return self._shape_cache[key]

    def _create_shape(self, kind: str, size: int) -> np.ndarray:
        """Create a shape array based on kind and size."""
        if kind == "square":
            return np.full((size, size), "wall", dtype="<U50")
        elif kind == "cross":
            s = size * 2 - 1
            out = np.full((s, s), "empty", dtype="<U50")
            mid = size - 1
            out[mid, :] = "wall"
            out[:, mid] = "wall"
            return out
        elif kind == "L":
            out = np.full((size, size), "empty", dtype="<U50")
            out[:, 0] = "wall"
            out[size - 1, :] = "wall"
            return out
        else:  # "block" or default
            return np.array([["wall"]], dtype="<U50")

    # ========== Utility Functions ==========

    def _get_inner_bounds(self, grid: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """Get bounds of the inner area (excluding border)."""
        bw = self._config.border_width
        h, w = grid.shape

        top, left = bw, bw
        bottom, right = h - bw - 1, w - bw - 1

        if bottom < top or right < left:
            return None

        return top, bottom, left, right

    def _get_object_bounds(self, grid: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """Get bounds for object placement (with margin for 3x3 neighborhood)."""
        bw = self._config.border_width
        h, w = grid.shape

        top, left = bw + 1, bw + 1
        bottom, right = h - bw - 2, w - bw - 2

        if bottom < top or right < left:
            return None

        return top, bottom, left, right

    @staticmethod
    def _dilate_bool(mask: np.ndarray, radius: int = 1) -> np.ndarray:
        """Fast Chebyshev dilation using boolean shifts."""
        h, w = mask.shape
        out = np.zeros_like(mask, dtype=bool)

        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                si0 = max(0, -di)
                ti0 = max(0, di)
                ih = h - abs(di)

                sj0 = max(0, -dj)
                tj0 = max(0, dj)
                jw = w - abs(dj)

                if ih > 0 and jw > 0:
                    out[ti0 : ti0 + ih, tj0 : tj0 + jw] |= mask[si0 : si0 + ih, sj0 : sj0 + jw]

        return out
