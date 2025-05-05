from typing import Dict, Optional, Tuple, Union

import numpy as np

from mettagrid.config.room.room import Room


class VariedTerrainObjects(Room):
    """
    A grid-world environment with varied terrain features and teams:
      - Large and small random obstacles
      - Cross-shaped obstacles
      - Mini labyrinths with thickened passages and altar scatter
      - Scattered walls
      - Rectangular blocks
      - Colored mines (red/green/blue)
      - Colored generators (red/green/blue)
      - Altars
      - Agents distributed across three teams

    Objects placed with at least one-cell clearance; skipped if no space.
    Build order:
        labyrinths → obstacles → scattered walls → blocks → mines → generators → altars → agents
    """

    STYLE_PARAMETERS = {
        "all-sparse": {
            "hearts_count": {"count": [3, 10]},
            "mines": {"red": {"count": [1, 3]}, "green": {"count": [1, 3]}, "blue": {"count": [1, 3]}},
            "generators": {"red": {"count": [1, 3]}, "green": {"count": [1, 3]}, "blue": {"count": [1, 3]}},
            "large_obstacles": {"size_range": [10, 25], "count": [0, 2]},
            "small_obstacles": {"size_range": [3, 6], "count": [0, 2]},
            "crosses": {"count": [0, 2]},
            "labyrinths": {"count": [0, 2]},
            "scattered_walls": {"count": [0, 2]},
            "blocks": {"count": [0, 2]},
            "clumpiness": [0, 2],
        },
        "balanced": {
            "hearts_count": {"count": [5, 20]},
            "mines": {"red": {"count": [5, 9]}, "green": {"count": [5, 9]}, "blue": {"count": [5, 9]}},
            "generators": {"red": {"count": [5, 9]}, "green": {"count": [5, 9]}, "blue": {"count": [5, 9]}},
            "large_obstacles": {"size_range": [10, 25], "count": [3, 7]},
            "small_obstacles": {"size_range": [3, 6], "count": [3, 7]},
            "crosses": {"count": [3, 7]},
            "labyrinths": {"count": [3, 7]},
            "scattered_walls": {"count": [3, 7]},
            "blocks": {"count": [3, 7]},
            "clumpiness": [1, 3],
        },
        "sparse-altars-dense-objects": {
            "hearts_count": {"count": [3, 10]},
            "mines": {"red": {"count": [3, 7]}, "green": {"count": [3, 7]}, "blue": {"count": [3, 7]}},
            "generators": {"red": {"count": [3, 7]}, "green": {"count": [3, 7]}, "blue": {"count": [3, 7]}},
            "large_obstacles": {"size_range": [10, 25], "count": [8, 15]},
            "small_obstacles": {"size_range": [3, 6], "count": [8, 15]},
            "crosses": {"count": [7, 15]},
            "labyrinths": {"count": [6, 15]},
            "scattered_walls": {"count": [40, 60]},
            "blocks": {"count": [5, 15]},
            "clumpiness": [2, 6],
        },
        "maze": {
            "hearts_count": {"count": [5, 15]},
            "mines": {"red": {"count": [2, 10]}, "green": {"count": [2, 10]}, "blue": {"count": [2, 10]}},
            "generators": {"red": {"count": [2, 10]}, "green": {"count": [2, 10]}, "blue": {"count": [2, 10]}},
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
        num_agents: Union[int, Dict[str, int]] = 3,
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: str = "wall",
        occupancy_threshold: float = 0.66,
        style: str = "balanced",
        teams: list | None = None,
        object_colors: list | None = None,
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self.labels.append(style)  # Add style to labels after parent init

        # Use provided dimensions
        self._width = width
        self._height = height
        self.set_size_labels(width, height)

        # RNG setup
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

        self._occupancy_threshold = occupancy_threshold

        if teams is None:
            self.teams = ["agent"]
        else:
            self.teams = teams

        if object_colors is None:
            self.object_colors = ["red"]
        else:
            self.object_colors = object_colors

        # Validate style
        if style not in self.STYLE_PARAMETERS:
            raise ValueError(f"Unknown style: '{style}'. Available: {list(self.STYLE_PARAMETERS.keys())}")
        params = self.STYLE_PARAMETERS[style]

        # Compute scaling
        area = width * height
        scale = area / 6000.0

        # Average sizes for occupancy clamping
        avg_sizes = {
            "large_obstacles": 17.5,
            "small_obstacles": 4.5,
            "crosses": 9,
            "labyrinths": 72,
            "scattered_walls": 1,
            "blocks": 64,
            "mines": 1,
            "generators": 1,
            "hearts": 1,
        }

        allowed_fraction = 0.3

        def clamp_count(range_vals, avg):
            if isinstance(range_vals, dict):
                range_vals = range_vals["count"]
            base = self._rng.integers(range_vals[0], range_vals[1] + 1)
            scaled = int(base * scale)
            max_allowed = int((allowed_fraction * area) / avg)
            return min(max(scaled, 0), max_allowed) if scaled > 0 else 0

        # Feature counts
        self._large_obstacles = {
            "size_range": params["large_obstacles"]["size_range"],
            "count": clamp_count(params["large_obstacles"]["count"], avg_sizes["large_obstacles"]),
        }
        self._small_obstacles = {
            "size_range": params["small_obstacles"]["size_range"],
            "count": clamp_count(params["small_obstacles"]["count"], avg_sizes["small_obstacles"]),
        }
        self._crosses = {"count": clamp_count(params["crosses"]["count"], avg_sizes["crosses"])}
        self._labyrinths = {"count": clamp_count(params["labyrinths"]["count"], avg_sizes["labyrinths"])}
        self._scattered_walls = {"count": clamp_count(params["scattered_walls"]["count"], avg_sizes["scattered_walls"])}
        self._blocks = {"count": clamp_count(params["blocks"]["count"], avg_sizes["blocks"])}

        # Initialize mines and generators with color-specific counts
        self._mines = {color: clamp_count(params["mines"][color], avg_sizes["mines"]) for color in self.object_colors}
        self._generators = {
            color: clamp_count(params["generators"][color], avg_sizes["generators"]) for color in self.object_colors
        }
        self._hearts_count = clamp_count(params["hearts_count"], avg_sizes["hearts"])

        # Initialize teams with equal distribution
        if isinstance(num_agents, int):
            total = num_agents
            base = total // 3
            rem = total % 3
            self._agent_counts = {f"team_{i + 1}": base + (1 if i < rem else 0) for i in range(3)}
        elif isinstance(num_agents, dict):
            self._agent_counts = num_agents.copy()  # Make a copy to avoid modifying input
        else:
            raise ValueError("agents must be int or dict of team counts")

    def _build(self) -> np.ndarray:
        # Initialize grid and occupancy mask
        grid = np.full((self._height, self._width), "empty", dtype=object)
        self._occupancy = np.zeros((self._height, self._width), dtype=bool)

        # Place in order
        for method in [
            self._place_labyrinths,
            self._place_all_obstacles,
            self._place_scattered_walls,
            self._place_blocks,
            self._place_mines,
            self._place_generators,
            self._place_altars,
            self._place_agents,
        ]:
            try:
                grid = method(grid)
            except Exception as e:
                print(f"Error in {method.__name__}: {str(e)}")
                raise

        return grid

    def _choose_random_empty(self) -> Optional[Tuple[int, int]]:
        """Find a random empty cell in the grid."""
        empty_cells = np.where(~self._occupancy)
        if len(empty_cells[0]) == 0:
            return None
        idx = self._rng.integers(len(empty_cells[0]))
        return empty_cells[0][idx], empty_cells[1][idx]

    def _place_mines(self, grid: np.ndarray) -> np.ndarray:
        """Place colored mines according to specified counts."""
        for color, count in self._mines.items():
            for _ in range(count):
                pos = self._choose_random_empty()
                if pos is None:
                    break
                r, c = pos
                grid[r, c] = f"mine.{color}"
                self._occupancy[r, c] = True
        return grid

    def _place_generators(self, grid: np.ndarray) -> np.ndarray:
        """Place colored generators according to specified counts."""
        for color, count in self._generators.items():
            for _ in range(count):
                pos = self._choose_random_empty()
                if pos is None:
                    break
                r, c = pos
                grid[r, c] = f"generator.{color}"
                self._occupancy[r, c] = True
        return grid

    def _place_altars(self, grid: np.ndarray) -> np.ndarray:
        """Place altars according to specified count."""
        current = np.count_nonzero(grid == "altar")
        for _ in range(self._hearts_count - current):
            pos = self._choose_random_empty()
            if pos is None:
                break
            r, c = pos
            grid[r, c] = "altar"
            self._occupancy[r, c] = True
        return grid

    def _place_agents(self, grid: np.ndarray) -> np.ndarray:
        """Place agents of different teams."""
        # Iterate through teams in order
        for team in self.teams:
            count = self._agent_counts.get(team, 0)
            for _ in range(count):
                pos = self._choose_random_empty()
                if pos is None:
                    break
                r, c = pos
                grid[r, c] = f"agent.{team}"
                self._occupancy[r, c] = True
        return grid

    def _place_all_obstacles(self, grid: np.ndarray) -> np.ndarray:
        """Place all types of obstacles."""
        # Place large obstacles
        for _ in range(self._large_obstacles["count"]):
            size = self._rng.integers(
                self._large_obstacles["size_range"][0], self._large_obstacles["size_range"][1] + 1
            )
            pos = self._choose_random_empty()
            if pos is None:
                break
            r, c = pos
            if r + size <= self._height and c + size <= self._width:
                grid[r : r + size, c : c + size] = "wall"
                self._occupancy[r : r + size, c : c + size] = True

        # Place small obstacles
        for _ in range(self._small_obstacles["count"]):
            size = self._rng.integers(
                self._small_obstacles["size_range"][0], self._small_obstacles["size_range"][1] + 1
            )
            pos = self._choose_random_empty()
            if pos is None:
                break
            r, c = pos
            if r + size <= self._height and c + size <= self._width:
                grid[r : r + size, c : c + size] = "wall"
                self._occupancy[r : r + size, c : c + size] = True

        # Place crosses
        for _ in range(self._crosses["count"]):
            pos = self._choose_random_empty()
            if pos is None:
                break
            r, c = pos
            # Create cross shape
            size = 3  # Fixed size for crosses
            if r + size <= self._height and c + size <= self._width:
                grid[r : r + size, c + size // 2] = "wall"
                grid[r + size // 2, c : c + size] = "wall"
                self._occupancy[r : r + size, c + size // 2] = True
                self._occupancy[r + size // 2, c : c + size] = True

        return grid

    def _place_scattered_walls(self, grid: np.ndarray) -> np.ndarray:
        """Place individual scattered walls."""
        for _ in range(self._scattered_walls["count"]):
            pos = self._choose_random_empty()
            if pos is None:
                break
            r, c = pos
            grid[r, c] = "wall"
            self._occupancy[r, c] = True
        return grid

    def _place_blocks(self, grid: np.ndarray) -> np.ndarray:
        """Place rectangular blocks."""
        for _ in range(self._blocks["count"]):
            size = self._rng.integers(4, 9)  # Random block size
            pos = self._choose_random_empty()
            if pos is None:
                break
            r, c = pos
            if r + size <= self._height and c + size <= self._width:
                grid[r : r + size, c : c + size] = "wall"
                self._occupancy[r : r + size, c : c + size] = True
        return grid

    def _place_labyrinths(self, grid: np.ndarray) -> np.ndarray:
        """Place mini labyrinths."""
        for _ in range(self._labyrinths["count"]):
            size = self._rng.integers(10, 30)  # Fixed size for labyrinths
            pos = self._choose_random_empty()
            if pos is None:
                break
            r, c = pos
            if r + size <= self._height and c + size <= self._width:
                # Create basic maze pattern
                for i in range(size):
                    for j in range(size):
                        if i % 2 == 0 or j % 2 == 0:
                            if self._rng.random() < 0.7:  # 70% chance of wall
                                grid[r + i, c + j] = "wall"
                                self._occupancy[r + i, c + j] = True
        return grid
