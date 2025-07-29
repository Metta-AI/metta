from dataclasses import dataclass
from typing import Literal

import numpy as np

from metta.common.util.config import Config
from metta.map.scene import Scene
from metta.map.types import MapGrid


class PerfectMazeParams(Config):
    """
    Parameters for perfect maze generation.

    A perfect maze has exactly one path between any two points, with no loops or inaccessible areas.
    This implementation forces one-tile hallways (room_size=1, wall_size=1) for true maze corridors.
    """
    algorithm: Literal["recursive_backtracking", "kruskal", "prim"] = "recursive_backtracking"

    # Optional: specify entrance and exit points
    entrance: Literal["top-left", "top-right", "bottom-left", "bottom-right"] | None = None
    exit: Literal["top-left", "top-right", "bottom-left", "bottom-right"] | None = None


class PerfectMaze(Scene[PerfectMazeParams]):
    """
    Perfect maze generator with one-tile hallways.

    A perfect maze is a maze where:
    1. Every cell is reachable from every other cell
    2. There is exactly one path between any two cells (no loops)
    3. No areas are inaccessible

    This implementation uses one-tile corridors separated by one-tile walls for classic maze appearance.

    Algorithms:
    - recursive_backtracking: Creates long, winding passages with deep branching
    - kruskal: More balanced tree structure, tends to have shorter dead ends
    - prim: Similar to Kruskal but grows from a single point, creating more organic shapes

    Example output (# = wall, . = empty):
    ┌─────────┐
    │#########│
    │#...#...#│
    │###.#.#.#│
    │#...#.#.#│
    │#.###.#.#│
    │#.....#.#│
    │#######.#│
    │.......#.│
    │#########│
    └─────────┘
    """

    def post_init(self):
        """Initialize the maze grid structure."""
        # Perfect mazes use 1-tile rooms and 1-tile walls for classic appearance
        self.room_size = 1
        self.wall_size = 1

        # Calculate maze dimensions (must be odd for proper maze structure)
        self.maze_width = (self.width + 1) // 2
        self.maze_height = (self.height + 1) // 2

        # Ensure we have odd dimensions for proper maze structure
        if self.width % 2 == 0:
            self.width -= 1
        if self.height % 2 == 0:
            self.height -= 1

    def _get_maze_neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        """Get valid neighboring cells in the maze grid (2 steps away)."""
        neighbors = []
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.maze_height and 0 <= new_col < self.maze_width:
                neighbors.append((new_row, new_col))
        return neighbors

    def _carve_passage(self, from_row: int, from_col: int, to_row: int, to_col: int):
        """Carve a passage between two maze cells, including the wall between them."""
        # Convert maze coordinates to grid coordinates
        start_y, start_x = from_row * 2, from_col * 2
        end_y, end_x = to_row * 2, to_col * 2

        # Carve the destination cell
        self.grid[end_y, end_x] = "empty"

        # Carve the wall between them
        wall_y = (start_y + end_y) // 2
        wall_x = (start_x + end_x) // 2
        self.grid[wall_y, wall_x] = "empty"

    def _render_recursive_backtracking(self):
        """
        Recursive backtracking algorithm.

        Creates mazes with long, winding corridors and deep branching.
        Tends to create mazes with a "river" pattern.
        """
        # Initialize grid with walls
        self.grid[:] = "wall"

        # Track visited cells in maze coordinates
        visited = np.zeros((self.maze_height, self.maze_width), dtype=bool)

        def carve_recursive(row: int, col: int):
            # Mark current cell as visited and carve it
            visited[row, col] = True
            grid_y, grid_x = row * 2, col * 2
            self.grid[grid_y, grid_x] = "empty"

            # Get neighbors and shuffle for randomness
            neighbors = self._get_maze_neighbors(row, col)
            self.rng.shuffle(neighbors)

            # Visit each unvisited neighbor
            for neighbor_row, neighbor_col in neighbors:
                if not visited[neighbor_row, neighbor_col]:
                    # Carve passage to neighbor
                    self._carve_passage(row, col, neighbor_row, neighbor_col)
                    carve_recursive(neighbor_row, neighbor_col)

        # Start from top-left corner
        carve_recursive(0, 0)

    def _render_kruskal(self):
        """
        Kruskal's algorithm using Union-Find.

        Creates more balanced mazes with shorter dead ends.
        Good balance between long corridors and branching.
        """
        # Initialize grid with walls
        self.grid[:] = "wall"

        # Create all possible walls between adjacent cells
        walls = []
        for row in range(self.maze_height):
            for col in range(self.maze_width):
                # Add wall to right neighbor
                if col + 1 < self.maze_width:
                    walls.append(((row, col), (row, col + 1)))
                # Add wall to bottom neighbor
                if row + 1 < self.maze_height:
                    walls.append(((row, col), (row + 1, col)))

        # Shuffle walls for randomness
        self.rng.shuffle(walls)

        # Union-Find data structure
        parent = {}
        rank = {}

        def make_set(cell):
            parent[cell] = cell
            rank[cell] = 0

        def find(cell):
            if parent[cell] != cell:
                parent[cell] = find(parent[cell])
            return parent[cell]

        def union(cell1, cell2):
            root1, root2 = find(cell1), find(cell2)
            if root1 != root2:
                if rank[root1] < rank[root2]:
                    parent[root1] = root2
                elif rank[root1] > rank[root2]:
                    parent[root2] = root1
                else:
                    parent[root2] = root1
                    rank[root1] += 1
                return True
            return False

        # Initialize all cells as separate sets
        for row in range(self.maze_height):
            for col in range(self.maze_width):
                make_set((row, col))
                # Carve the cell itself
                grid_y, grid_x = row * 2, col * 2
                self.grid[grid_y, grid_x] = "empty"

        # Process walls in random order
        for (cell1, cell2) in walls:
            if union(cell1, cell2):
                # Carve passage between cells
                self._carve_passage(cell1[0], cell1[1], cell2[0], cell2[1])

    def _render_prim(self):
        """
        Prim's algorithm for maze generation.

        Grows the maze from a single starting point, creating organic shapes.
        Tends to create mazes with shorter dead ends than recursive backtracking.
        """
        # Initialize grid with walls
        self.grid[:] = "wall"

        # Track which cells are part of the maze
        in_maze = np.zeros((self.maze_height, self.maze_width), dtype=bool)

        # Start with a random cell
        start_row = self.rng.integers(0, self.maze_height)
        start_col = self.rng.integers(0, self.maze_width)

        # Add starting cell to maze
        in_maze[start_row, start_col] = True
        grid_y, grid_x = start_row * 2, start_col * 2
        self.grid[grid_y, grid_x] = "empty"

        # Track frontier walls (walls between maze and non-maze cells)
        frontier = []
        for neighbor in self._get_maze_neighbors(start_row, start_col):
            if not in_maze[neighbor[0], neighbor[1]]:
                frontier.append(((start_row, start_col), neighbor))

        while frontier:
            # Pick a random frontier wall
            wall_idx = self.rng.integers(0, len(frontier))
            maze_cell, frontier_cell = frontier.pop(wall_idx)

            # If frontier cell is not yet in maze, add it
            if not in_maze[frontier_cell[0], frontier_cell[1]]:
                # Add frontier cell to maze
                in_maze[frontier_cell[0], frontier_cell[1]] = True

                # Carve passage
                self._carve_passage(maze_cell[0], maze_cell[1], frontier_cell[0], frontier_cell[1])

                # Add new frontier walls
                for neighbor in self._get_maze_neighbors(frontier_cell[0], frontier_cell[1]):
                    if not in_maze[neighbor[0], neighbor[1]]:
                        # Check if this wall is already in frontier
                        wall = (frontier_cell, neighbor)
                        if wall not in frontier and (neighbor, frontier_cell) not in frontier:
                            frontier.append(wall)

    def _create_entrance_exit(self):
        """Create entrance and exit points if specified."""
        def get_position(anchor: str) -> tuple[int, int]:
            if anchor == "top-left":
                return (0, 0)
            elif anchor == "top-right":
                return (0, self.width - 1)
            elif anchor == "bottom-left":
                return (self.height - 1, 0)
            elif anchor == "bottom-right":
                return (self.height - 1, self.width - 1)

        if self.params.entrance:
            y, x = get_position(self.params.entrance)
            self.grid[y, x] = "empty"
            self.make_area(x, y, 1, 1, tags=["entrance", self.params.entrance])

        if self.params.exit:
            y, x = get_position(self.params.exit)
            self.grid[y, x] = "empty"
            self.make_area(x, y, 1, 1, tags=["exit", self.params.exit])

    def render(self):
        """Generate the perfect maze using the specified algorithm."""
        if self.params.algorithm == "recursive_backtracking":
            self._render_recursive_backtracking()
        elif self.params.algorithm == "kruskal":
            self._render_kruskal()
        elif self.params.algorithm == "prim":
            self._render_prim()
        else:
            raise ValueError(f"Unknown algorithm: {self.params.algorithm}")

        # Create entrance and exit points if specified
        self._create_entrance_exit()

        # Create corner areas for potential spawn points or special locations
        corner_positions = [
            ("top-left", 0, 0),
            ("top-right", self.width - 1, 0),
            ("bottom-left", 0, self.height - 1),
            ("bottom-right", self.width - 1, self.height - 1),
        ]

        for anchor, x, y in corner_positions:
            if self.grid[y, x] == "empty":  # Only create area if it's accessible
                self.make_area(x, y, 1, 1, tags=[anchor])
