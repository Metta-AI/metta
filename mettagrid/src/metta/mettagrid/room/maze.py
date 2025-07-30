import random

from metta.mettagrid.room.room import Room
from metta.mettagrid.room.utils import create_grid, set_position


# Maze generation using Prim's algorithm
class MazePrim(Room):
    EMPTY, WALL = "empty", "wall"
    START, END = "agent.agent", "altar"
    DIRECTIONS = [(2, 0), (-2, 0), (0, 2), (0, -2)]

    def __init__(
        self, width, height, start_pos, end_pos, branching=0.0, seed=None, border_width=0, border_object="wall"
    ):
        super().__init__(border_width=border_width, border_object=border_object, labels=["maze"])
        self._rng = random.Random(seed)
        self._width = width if width % 2 == 1 else width - 1
        self._height = height if height % 2 == 1 else height - 1
        self._start_pos = (set_position(start_pos[0], self._width), set_position(start_pos[1], self._height))
        self._end_pos = (set_position(end_pos[0], self._width), set_position(end_pos[1], self._height))
        self.set_size_labels(width, height)

    def _build(self):
        final_maze = create_grid(self._height, self._width, fill_value=self.WALL)
        maze = create_grid(self._height, self._width, fill_value=self.WALL)
        sx, sy = self._start_pos
        maze[sy, sx] = self.EMPTY
        walls = []
        for dx, dy in MazePrim.DIRECTIONS:
            wx, wy = sx + dx // 2, sy + dy // 2
            nx, ny = sx + dx, sy + dy
            if 0 <= nx < self._width and 0 <= ny < self._height:
                walls.append((wx, wy, nx, ny))
        while walls:
            idx = self._rng.randrange(len(walls))
            wx, wy, nx, ny = walls.pop(idx)
            if maze[ny, nx] == self.WALL:
                maze[wy, wx] = self.EMPTY
                maze[ny, nx] = self.EMPTY
                for dx, dy in MazePrim.DIRECTIONS:
                    nwx, nwy = nx + dx // 2, ny + dy // 2
                    nnx, nny = nx + dx, ny + dy
                    if 0 <= nnx < self._width and 0 <= nny < self._height and maze[nny, nnx] == self.WALL:
                        walls.append((nwx, nwy, nnx, nny))
        maze[self._start_pos[1], self._start_pos[0]] = self.START
        maze[self._end_pos[1], self._end_pos[0]] = self.END
        final_maze[: self._height, : self._width] = maze
        return final_maze


# Maze generation using Randomized Kruskal's algorithm
class MazeKruskal(Room):
    EMPTY, WALL = "empty", "wall"
    START, END = "agent.agent", "altar"

    def __init__(
        self, width, height, start_pos, end_pos, branching=0.0, seed=None, border_width=0, border_object="wall"
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._rng = random.Random(seed)
        self._width = width if width % 2 == 1 else width - 1
        self._height = height if height % 2 == 1 else height - 1
        self._start_pos = (set_position(start_pos[0], self._width), set_position(start_pos[1], self._height))
        self._end_pos = (set_position(end_pos[0], self._width), set_position(end_pos[1], self._height))

    def _build(self):
        final_maze = create_grid(self._height, self._width, fill_value=self.WALL)
        maze = create_grid(self._height, self._width, fill_value=self.WALL)
        cells = [(x, y) for y in range(1, self._height, 2) for x in range(1, self._width, 2)]
        for x, y in cells:
            maze[y, x] = self.EMPTY

        parent = {cell: cell for cell in cells}

        def find(cell):
            if parent[cell] != cell:
                parent[cell] = find(parent[cell])
            return parent[cell]

        def union(c1, c2):
            parent[find(c2)] = find(c1)

        walls = []
        for x, y in cells:
            for dx, dy in [(2, 0), (0, 2)]:
                nx, ny = x + dx, y + dy
                if nx < self._width and ny < self._height:
                    wx, wy = (x + nx) // 2, (y + ny) // 2
                    walls.append(((x, y), (nx, ny), (wx, wy)))
        self._rng.shuffle(walls)
        for cell1, cell2, wall in walls:
            if find(cell1) != find(cell2):
                wx, wy = wall
                maze[wy, wx] = self.EMPTY
                union(cell1, cell2)
        sx, sy = self._start_pos
        ex, ey = self._end_pos
        maze[sy, sx] = self.START
        maze[ey, ex] = self.END
        final_maze[: self._height, : self._width] = maze
        return final_maze
