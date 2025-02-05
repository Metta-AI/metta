import random
import numpy as np

from mettagrid.config.room.room import Room

def make_odd(x):
    if x % 2 == 0:
        x += 1
    return x

def set_position(x, upper_bound):
    """move within bounds, ensuring that the position is odd"""
    x = make_odd(x)
    if x < 0:
        x = 1
    elif x >= upper_bound:
        if x % 2 == 0:
            x = upper_bound - 1
        else:
            x = upper_bound - 2
    return x


class MazeDFS(Room):
    EMPTY, WALL = "empty", "wall"
    START, END = "agent.agent", "altar"
    NORTH, SOUTH, EAST, WEST = 'n', 's', 'e', 'w'

    def __init__(self, width, height, start_pos, end_pos, branching, seed=None, border_width=0, border_object="wall"):
        super().__init__(border_width=border_width, border_object=border_object)
        #self._width = width
        self._start_pos = start_pos
        self._end_pos = end_pos
        self._branching = branching
        self._rng = random.Random(seed)
        self.final_maze = np.full((height, width), self.WALL, dtype='<U50')

        # Validate branching parameter
        assert 0 <= self._branching <= 1, "Branching parameter must be between 0 and 1"

        # Compute an effective maze size that is odd.
        self._width = width if width % 2 == 1 else width - 1
        self._height = height if height % 2 == 1 else height - 1

        # The start and end positions must be odd and within the effective maze dimensions.
        self._start_pos = (set_position(self._start_pos[0], self._width), set_position(self._start_pos[1], self._height))
        self._end_pos = (set_position(self._end_pos[0], self._width), set_position(self._end_pos[1], self._height))

    def _build(self):
        # Generate the effective maze using the computed odd dimensions.
        maze = np.full((self._height, self._width), self.WALL, dtype='<U50')

        def should_branch():
            return self._rng.random() < self._branching

        def get_preferred_direction(x, y, target_x, target_y):
            if abs(target_x - x) > abs(target_y - y):
                return self.EAST if target_x > x else self.WEST
            return self.SOUTH if target_y > y else self.NORTH

        def visit(x, y, has_visited, target_x=None, target_y=None):
            maze[y, x] = self.EMPTY
            while True:
                unvisited_neighbors = []
                if y > 1 and (x, y - 2) not in has_visited:
                    unvisited_neighbors.append(self.NORTH)
                if y < self._height - 2 and (x, y + 2) not in has_visited:
                    unvisited_neighbors.append(self.SOUTH)
                if x > 1 and (x - 2, y) not in has_visited:
                    unvisited_neighbors.append(self.WEST)
                if x < self._width - 2 and (x + 2, y) not in has_visited:
                    unvisited_neighbors.append(self.EAST)

                if not unvisited_neighbors:
                    return

                # If a target is provided and we are not branching,
                # try to move in the preferred direction.
                if target_x is not None and target_y is not None and not should_branch():
                    preferred = get_preferred_direction(x, y, target_x, target_y)
                    if preferred in unvisited_neighbors:
                        next_direction = preferred
                    else:
                        next_direction = self._rng.choice(unvisited_neighbors)
                else:
                    next_direction = self._rng.choice(unvisited_neighbors)

                next_x, next_y = x, y
                if next_direction == self.NORTH:
                    next_x, next_y = x, y - 2
                    maze[y - 1, x] = self.EMPTY
                elif next_direction == self.SOUTH:
                    next_x, next_y = x, y + 2
                    maze[y + 1, x] = self.EMPTY
                elif next_direction == self.WEST:
                    next_x, next_y = x - 2, y
                    maze[y, x - 1] = self.EMPTY
                elif next_direction == self.EAST:
                    next_x, next_y = x + 2, y
                    maze[y, x + 1] = self.EMPTY

                # Optionally create a branch by clearing an alternate neighbor.
                if should_branch() and len(unvisited_neighbors) > 1:
                    alt_directions = [d for d in unvisited_neighbors if d != next_direction]
                    alt_direction = self._rng.choice(alt_directions)
                    if alt_direction == self.NORTH:
                        maze[y - 1, x] = self.EMPTY
                    elif alt_direction == self.SOUTH:
                        maze[y + 1, x] = self.EMPTY
                    elif alt_direction == self.WEST:
                        maze[y, x - 1] = self.EMPTY
                    elif alt_direction == self.EAST:
                        maze[y, x + 1] = self.EMPTY

                has_visited.append((next_x, next_y))
                visit(next_x, next_y, has_visited,
                      target_x=target_x,
                      target_y=target_y)

        # Start the maze generation from the start position.
        has_visited = [self._start_pos]
        visit(self._start_pos[0], self._start_pos[1], has_visited,
              target_x=self._end_pos[0] if self._end_pos else None,
              target_y=self._end_pos[1] if self._end_pos else None)

        # Mark start and end positions in the effective maze.
        maze[self._start_pos[1], self._start_pos[0]] = self.START
        if self._end_pos:
            maze[self._end_pos[1], self._end_pos[0]] = self.END

        # Create the final maze with the full container dimensions,
        # initially filled with walls.
        # Copy the effective maze into the top-left corner of the container.
        self.final_maze[:self._height, :self._width] = maze

        return self.final_maze


class MazePrim(Room):
    EMPTY, WALL = "empty", "wall"
    START, END = "agent.agent", "altar"
    # Directions defined as moves of 2 cells (to jump over a wall)
    DIRECTIONS = [(2, 0), (-2, 0), (0, 2), (0, -2)]

    def __init__(self, width, height, start_pos, end_pos, branching=0.0, seed=None, border_width=0, border_object="wall"):
        """
        branching parameter is kept for API compatibility; it isn't used in Prim's algorithm.
        """
        super().__init__(border_width=border_width, border_object=border_object)
        self._rng = random.Random(seed)
        # The final container retains the provided dimensions.
        self.final_maze = np.full((height, width), self.WALL, dtype='<U50')
        # Compute an effective maze size that is odd.
        self._width = width if width % 2 == 1 else width - 1
        self._height = height if height % 2 == 1 else height - 1

        # Adjust the start and end positions to be odd and within the effective maze dimensions.
        self._start_pos = (set_position(start_pos[0], self._width), set_position(start_pos[1], self._height))
        self._end_pos = (set_position(end_pos[0], self._width), set_position(end_pos[1], self._height))

    def _build(self) -> np.ndarray:
        """
        Generate a maze using a version of Prim's algorithm:
          - We start at the (adjusted) start position.
          - The algorithm carves passages by maintaining a list of walls.
          - Once finished, the start and end markers are applied.
          - The effective maze is copied into the final container.
        """
        maze = np.full((self._height, self._width), self.WALL, dtype='<U50')

        # Start at the given start position.
        sx, sy = self._start_pos
        maze[sy, sx] = self.EMPTY

        walls = []
        # Add neighboring walls from the start cell.
        for dx, dy in MazePrim.DIRECTIONS:
            wx, wy = sx + dx // 2, sy + dy // 2
            nx, ny = sx + dx, sy + dy
            if 0 <= nx < self._width and 0 <= ny < self._height:
                walls.append((wx, wy, nx, ny))

        # Process the wall list.
        while walls:
            idx = self._rng.randrange(len(walls))
            wx, wy, nx, ny = walls.pop(idx)
            if maze[ny, nx] == self.WALL:
                # Carve through the wall.
                maze[wy, wx] = self.EMPTY
                maze[ny, nx] = self.EMPTY
                # Add the neighboring walls of the newly carved cell.
                for dx, dy in MazePrim.DIRECTIONS:
                    nwx, nwy = nx + dx // 2, ny + dy // 2
                    nnx, nny = nx + dx, ny + dy
                    if 0 <= nnx < self._width and 0 <= nny < self._height:
                        if maze[nny, nnx] == self.WALL:
                            walls.append((nwx, nwy, nnx, nny))

        # Place the start and end markers.
        maze[self._start_pos[1], self._start_pos[0]] = self.START
        maze[self._end_pos[1], self._end_pos[0]] = self.END

        # Copy the effective maze into the full container.
        self.final_maze[:self._height, :self._width] = maze

        return self.final_maze

class MazeKruskal(Room):
    EMPTY, WALL = "empty", "wall"
    START, END = "agent.agent", "altar"

    def __init__(self, width, height, start_pos, end_pos, branching=0.0, seed=None, border_width=0, border_object="wall"):
        """
        branching parameter is kept for API compatibility; it isn't used in Kruskal's algorithm.
        """
        super().__init__(border_width=border_width, border_object=border_object)
        self._rng = random.Random(seed)
        # The final container retains the provided dimensions.
        self.final_maze = np.full((height, width), self.WALL, dtype='<U50')
        # Compute an effective maze size that is odd.
        self._width = width if width % 2 == 1 else width - 1
        self._height = height if height % 2 == 1 else height - 1

        # Adjust start and end positions to be odd and within effective bounds.
        self._start_pos = (set_position(start_pos[0], self._width), set_position(start_pos[1], self._height))
        self._end_pos = (set_position(end_pos[0], self._width), set_position(end_pos[1], self._height))

    def _build(self) -> np.ndarray:
        """
        Generate a maze using Randomized Kruskal's algorithm.
          - Each cell (located at odd indices) is initialized as a passage.
          - Walls between adjacent cells are considered in random order.
          - If the wall separates two different sets (cells not yet connected),
            the wall is removed and the sets are merged.
          - The start and end markers are then applied.
          - The effective maze is copied into the final container.
        """
        # Initialize the maze: passages (cells) at odd indices and walls everywhere else.
        maze = np.full((self._height, self._width), self.WALL, dtype='<U50')
        cells = [(x, y) for y in range(1, self._height, 2) for x in range(1, self._width, 2)]
        for (x, y) in cells:
            maze[y, x] = self.EMPTY

        # Initialize disjoint-set (union-find) for each cell.
        parent = {}
        def find(cell):
            if parent[cell] != cell:
                parent[cell] = find(parent[cell])
            return parent[cell]

        def union(cell1, cell2):
            root1 = find(cell1)
            root2 = find(cell2)
            parent[root2] = root1

        # Each cell is its own set.
        for cell in cells:
            parent[cell] = cell

        # List all walls between adjacent cells.
        # Only consider rightward and downward walls to avoid duplicates.
        walls = []
        for (x, y) in cells:
            for dx, dy in [(2, 0), (0, 2)]:
                nx, ny = x + dx, y + dy
                if nx < self._width and ny < self._height:
                    # The wall is between (x, y) and (nx, ny).
                    wx, wy = (x + nx) // 2, (y + ny) // 2
                    walls.append(((x, y), (nx, ny), (wx, wy)))

        # Shuffle the walls.
        self._rng.shuffle(walls)

        # Process each wall in random order.
        for cell1, cell2, wall in walls:
            if find(cell1) != find(cell2):
                # Remove the wall.
                wx, wy = wall
                maze[wy, wx] = self.EMPTY
                union(cell1, cell2)

        # Place the start and end markers.
        sx, sy = self._start_pos
        ex, ey = self._end_pos
        maze[sy, sx] = self.START
        maze[ey, ex] = self.END

        # Copy the effective maze into the final container.
        self.final_maze[:self._height, :self._width] = maze
        return self.final_maze