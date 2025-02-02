import random

import numpy as np

from mettagrid.config.room.room import Room

class Maze(Room):
    EMPTY, WALL = "empty", "wall"
    START, END = "agent", "altar"
    NORTH, SOUTH, EAST, WEST = 'n', 's', 'e', 'w'

    def __init__(self, width, height, start_pos, end_pos, branching, seed=None, border_width=0, border_object="wall"):
        super().__init__(border_width=border_width, border_object=border_object)
        self._width = width
        self._height = height
        self._start_pos = start_pos
        self._end_pos = end_pos
        self._branching = branching
        self._rng = random.Random(seed)

        # Validate inputs
        assert 0 <= self._branching <= 1, "Branching parameter must be between 0 and 1"
        assert self._width % 2 == 1 and self._width >= 3, "Width must be odd and >= 3. Got {}".format(self._width)
        assert self._height % 2 == 1 and self._height >= 3, "Height must be odd and >= 3. Got {}".format(self._height)
        assert self._start_pos[0] % 2 == 1 and self._start_pos[1] % 2 == 1, "Start position must have odd coordinates"
        if self._end_pos:
            assert self._end_pos[0] % 2 == 1 and self._end_pos[1] % 2 == 1, "End position must have odd coordinates"
            assert 0 < self._end_pos[0] < self._width and 0 < self._end_pos[1] < self._height, "End position must be within maze bounds"
        assert 0 < self._start_pos[0] < self._width and 0 < self._start_pos[1] < self._height, "Start position must be within maze bounds"

    def _build(self):
        """
        Generate a maze and return it as a numpy array of characters,
        matching the format from build_map_from_ascii.
        """

        # Initialize numpy array directly
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

                if target_x and target_y and not should_branch():
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

                if should_branch() and len(unvisited_neighbors) > 1:
                    alt_direction = self._rng.choice([d for d in unvisited_neighbors if d != next_direction])
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
                    target_x=self._end_pos[0] if self._end_pos else None,
                    target_y=self._end_pos[1] if self._end_pos else None)

        has_visited = [self._start_pos]
        visit(self._start_pos[0], self._start_pos[1], has_visited,
            target_x=self._end_pos[0] if self._end_pos else None,
            target_y=self._end_pos[1] if self._end_pos else None)

        # Set start and end positions
        maze[self._start_pos[1], self._start_pos[0]] = self.START
        if self._end_pos:
            maze[self._end_pos[1], self._end_pos[0]] = self.END

        return maze
