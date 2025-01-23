from mettagrid.config.room_builder import MettaGridRoomBuilder
import numpy as np
import random

class MazeRoom(MettaGridRoomBuilder):
    EMPTY, WALL = ' ', 'W'
    START, END = 'A', 'a'
    NORTH, SOUTH, EAST, WEST = 'n', 's', 'e', 'w'

    def __init__(self, width, height, start_pos, end_pos, branching, seed=None):
        self.width = width
        self.height = height
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.branching = branching
        self.seed = seed
        
        # Validate inputs
        assert 0 <= self.branching <= 1, "Branching parameter must be between 0 and 1"
        assert self.width % 2 == 1 and self.width >= 3, "Width must be odd and >= 3"
        assert self.height % 2 == 1 and self.height >= 3, "Height must be odd and >= 3"
        assert self.start_pos[0] % 2 == 1 and self.start_pos[1] % 2 == 1, "Start position must have odd coordinates"
        if self.end_pos:
            assert self.end_pos[0] % 2 == 1 and self.end_pos[1] % 2 == 1, "End position must have odd coordinates"
            assert 0 < self.end_pos[0] < self.width and 0 < self.end_pos[1] < self.height, "End position must be within maze bounds"
        assert 0 < self.start_pos[0] < self.width and 0 < self.start_pos[1] < self.height, "Start position must be within maze bounds"

        if self.seed is not None:
            random.seed(self.seed)

    def build_room(self):
        # Generate the maze dictionary
        return self.create_maze()
        
    def create_maze(self):
        """
        Generate a maze and return it as a numpy array of characters,
        matching the format from build_map_from_ascii.
        """

        # Initialize numpy array directly
        maze = np.full((self.height, self.width), self.WALL, dtype=str)

        def should_branch():
            return random.random() < self.branching

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
                if y < self.height - 2 and (x, y + 2) not in has_visited:
                    unvisited_neighbors.append(self.SOUTH)
                if x > 1 and (x - 2, y) not in has_visited:
                    unvisited_neighbors.append(self.WEST)
                if x < self.width - 2 and (x + 2, y) not in has_visited:
                    unvisited_neighbors.append(self.EAST)

                if not unvisited_neighbors:
                    return

                if target_x and target_y and not should_branch():
                    preferred = get_preferred_direction(x, y, target_x, target_y)
                    if preferred in unvisited_neighbors:
                        next_direction = preferred
                    else:
                        next_direction = random.choice(unvisited_neighbors)
                else:
                    next_direction = random.choice(unvisited_neighbors)

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

                if should_branch() and len(unvisited_neighbors) > 1:
                    alt_direction = random.choice([d for d in unvisited_neighbors if d != next_direction])
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
                    target_x=self.end_pos[0] if self.end_pos else None,
                    target_y=self.end_pos[1] if self.end_pos else None)

        has_visited = [self.start_pos]
        visit(self.start_pos[0], self.start_pos[1], has_visited,
            target_x=self.end_pos[0] if self.end_pos else None,
            target_y=self.end_pos[1] if self.end_pos else None)

        # Set start and end positions
        maze[self.start_pos[1], self.start_pos[0]] = self.START
        if self.end_pos:
            maze[self.end_pos[1], self.end_pos[0]] = self.END

        return maze