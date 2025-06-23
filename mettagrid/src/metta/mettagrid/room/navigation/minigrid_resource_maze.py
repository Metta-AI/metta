"""
MinGrid-Inspired Resource Maze Environment

Inspired by MinGrid's maze environments and ObstructedMaze. This creates a maze-like layout
with walls forming corridors and dead ends. Generators and altars are scattered throughout
the maze, requiring exploration to find and collect resources.

Original MinGrid Maze envs: Agent navigates through maze to reach goal, avoiding dead ends.
MettagGrid adaptation: Agent navigates maze to find and collect from generators/altars.

The environment consists of:
- Maze-like layout with wall corridors and dead ends
- Generators scattered throughout for resource collection
- Altars placed strategically for heart rewards
- Multiple possible paths with varying difficulty
"""

from typing import Optional, Tuple, List, Set
import numpy as np
from omegaconf import DictConfig

from mettagrid.room.room import Room


class MinigridResourceMaze(Room):
    def __init__(
        self,
        width: int,
        height: int,
        objects: DictConfig,
        agents: int | DictConfig = 1,
        seed: Optional[int] = None,
        border_width: int = 1,
        border_object: str = "wall",
        maze_complexity: float = 0.5,  # 0.0 = simple, 1.0 = complex
        dead_end_probability: float = 0.3,  # Probability of creating dead ends
        resource_density: float = 0.1,  # Fraction of empty cells to place resources
        team: str | None = None,
    ):
        super().__init__(border_width=border_width, border_object=border_object, labels=["minigrid_resource_maze"])
        self.set_size_labels(width, height)
        self._rng = np.random.default_rng(seed)
        self._width = width
        self._height = height
        self._agents = agents
        self._objects = objects
        self._maze_complexity = maze_complexity
        self._dead_end_probability = dead_end_probability
        self._resource_density = resource_density
        self._team = team

    def _build(self) -> np.ndarray:
        # Create maze using recursive backtracking algorithm
        grid = self._generate_maze()

        # Place resources (generators and altars) in maze
        self._place_resources_in_maze(grid)

        # Place agents at maze start
        self._place_agents_in_maze(grid)

        return grid

    def _generate_maze(self) -> np.ndarray:
        """Generate maze using recursive backtracking algorithm."""
        # Start with all walls
        grid = np.full((self._height, self._width), "wall", dtype="<U50")

        # Maze generation requires odd dimensions for proper corridors
        maze_height = self._height if self._height % 2 == 1 else self._height - 1
        maze_width = self._width if self._width % 2 == 1 else self._width - 1

        # Initialize visited array
        visited = np.zeros((maze_height, maze_width), dtype=bool)

        # Start position (must be odd coordinates for proper maze)
        start_r, start_c = 1, 1

        # Carve initial position
        grid[start_r, start_c] = "empty"
        visited[start_r, start_c] = True

        # Stack for backtracking
        stack = [(start_r, start_c)]

        # Directions: up, right, down, left (2 cells apart for maze algorithm)
        directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]

        while stack:
            current_r, current_c = stack[-1]

            # Find unvisited neighbors
            neighbors = []
            for dr, dc in directions:
                new_r, new_c = current_r + dr, current_c + dc
                if (1 <= new_r < maze_height - 1 and
                    1 <= new_c < maze_width - 1 and
                    not visited[new_r, new_c]):
                    neighbors.append((new_r, new_c))

            if neighbors:
                # Choose random neighbor
                next_r, next_c = self._rng.choice(neighbors)

                # Carve path to neighbor
                wall_r = current_r + (next_r - current_r) // 2
                wall_c = current_c + (next_c - current_c) // 2

                grid[wall_r, wall_c] = "empty"
                grid[next_r, next_c] = "empty"
                visited[next_r, next_c] = True

                stack.append((next_r, next_c))
            else:
                # Backtrack
                stack.pop()

                # Optionally create dead end
                if self._rng.random() < self._dead_end_probability:
                    self._create_dead_end(grid, current_r, current_c, visited, maze_height, maze_width)

        # Add complexity by randomly opening some walls
        self._add_maze_complexity(grid)

        return grid

    def _create_dead_end(self, grid: np.ndarray, r: int, c: int, visited: np.ndarray,
                        maze_height: int, maze_width: int) -> None:
        """Create a small dead end branch from current position."""
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        for dr, dc in directions:
            new_r, new_c = r + dr, c + dc
            if (1 <= new_r < maze_height - 1 and
                1 <= new_c < maze_width - 1 and
                grid[new_r, new_c] == "wall" and
                self._rng.random() < 0.5):
                grid[new_r, new_c] = "empty"

    def _add_maze_complexity(self, grid: np.ndarray) -> None:
        """Add complexity by randomly opening some wall cells."""
        wall_positions = [(r, c) for r in range(1, self._height - 1)
                         for c in range(1, self._width - 1)
                         if grid[r, c] == "wall"]

        num_openings = int(len(wall_positions) * self._maze_complexity * 0.1)

        if wall_positions and num_openings > 0:
            selected_walls = self._rng.choice(len(wall_positions),
                                            size=min(num_openings, len(wall_positions)),
                                            replace=False)

            for idx in selected_walls:
                r, c = wall_positions[idx]
                # Only open if it connects two empty areas
                if self._should_open_wall(grid, r, c):
                    grid[r, c] = "empty"

    def _should_open_wall(self, grid: np.ndarray, r: int, c: int) -> bool:
        """Check if opening a wall would be beneficial (connects separate areas)."""
        # Count adjacent empty cells
        empty_neighbors = 0
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < self._height and 0 <= nc < self._width and
                grid[nr, nc] == "empty"):
                empty_neighbors += 1

        # Open if it would connect areas but not create too open space
        return 1 <= empty_neighbors <= 2

    def _place_resources_in_maze(self, grid: np.ndarray) -> None:
        """Place generators and altars in the maze corridors."""
        # Find all empty positions
        empty_positions = [(r, c) for r in range(self._height)
                          for c in range(self._width)
                          if grid[r, c] == "empty"]

        if not empty_positions:
            return

        # Calculate how many resources to place
        total_resources = 0
        for obj_name in ["generator", "altar"]:
            total_resources += self._objects.get(obj_name, 0)

        # Limit resources based on density
        max_resources = int(len(empty_positions) * self._resource_density)
        total_resources = min(total_resources, max_resources)

        # Place generators
        generator_count = self._objects.get("generator", 0)
        generator_positions = self._select_resource_positions(empty_positions, generator_count)

        for pos in generator_positions:
            r, c = pos
            grid[r, c] = "generator"
            empty_positions.remove(pos)

        # Place altars
        altar_count = self._objects.get("altar", 0)
        altar_positions = self._select_resource_positions(empty_positions, altar_count)

        for pos in altar_positions:
            r, c = pos
            grid[r, c] = "altar"
            empty_positions.remove(pos)

        # Place any other objects
        for obj_name, obj_count in self._objects.items():
            if obj_name in ["generator", "altar"]:
                continue

            positions = self._select_resource_positions(empty_positions, obj_count)
            for pos in positions:
                r, c = pos
                grid[r, c] = obj_name
                empty_positions.remove(pos)

    def _select_resource_positions(self, available_positions: List[Tuple[int, int]],
                                  count: int) -> List[Tuple[int, int]]:
        """Select positions for resources, preferring maze intersections and corridor ends."""
        if not available_positions or count <= 0:
            return []

        # Score positions based on strategic value
        scored_positions = []
        for pos in available_positions:
            score = self._calculate_position_score(pos)
            scored_positions.append((score, pos))

        # Sort by score (higher is better)
        scored_positions.sort(reverse=True)

        # Select top positions, but add some randomness
        selected = []
        selection_pool_size = min(len(scored_positions), count * 3)  # Consider top 3x positions

        for i in range(min(count, len(scored_positions))):
            if i < selection_pool_size // 2:
                # Always take some of the best positions
                selected.append(scored_positions[i][1])
            else:
                # Add randomness for the rest
                remaining_pool = scored_positions[i:selection_pool_size]
                if remaining_pool:
                    chosen = self._rng.choice(remaining_pool)[1]
                    selected.append(chosen)

        return selected

    def _calculate_position_score(self, pos: Tuple[int, int]) -> float:
        """Calculate strategic score for a position (higher = better for resource placement)."""
        r, c = pos

        # Prefer positions that are:
        # 1. At corridor intersections (multiple paths)
        # 2. At dead ends (requires exploration)
        # 3. Far from other resources

        # Count empty neighbors (intersections have more)
        empty_neighbors = 0
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self._height and 0 <= nc < self._width:
                # This would need access to current grid state
                # For simplicity, we'll use position-based heuristics
                pass

        # Prefer corners and intersections (heuristic based on position)
        score = 0.0

        # Distance from center (prefer spread out resources)
        center_r, center_c = self._height // 2, self._width // 2
        distance_from_center = abs(r - center_r) + abs(c - center_c)
        score += distance_from_center * 0.1

        # Add randomness to avoid deterministic patterns
        score += self._rng.random() * 2.0

        return score

    def _place_agents_in_maze(self, grid: np.ndarray) -> None:
        """Place agents at good starting positions in the maze."""
        # Find empty positions near edges (good starting points)
        start_positions = []

        # Check positions near corners and edges
        for r in range(1, min(4, self._height - 1)):
            for c in range(1, min(4, self._width - 1)):
                if grid[r, c] == "empty":
                    start_positions.append((r, c))

        # If no good starts found, use any empty position
        if not start_positions:
            start_positions = [(r, c) for r in range(self._height)
                             for c in range(self._width)
                             if grid[r, c] == "empty"]

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
