from typing import Set, Tuple, Union, List
import numpy as np
import math
from omegaconf import DictConfig

from mettagrid.config.room.room import Room


class RadialMaze(Room):
    """
    A generalizable radial maze map with a central starting position.
    
    This room is generated on a grid initially filled with walls. From the center,
    corridors (arms) radiate outward. You can specify:
      - the number of arms (between 4 and 12),
      - the length of each arm, and
      - the width of each arm (in grid cells).
      
    Special objects are placed at the end of the first three arms:
      - Arm 0: generator
      - Arm 1: converter
      - Arm 2: heart altar
      
    The agent is placed at the center.
    """

    def __init__(
        self,
        width: int,
        height: int,
        radial_params: DictConfig,
        seed: Union[int, None] = None,
        border_width: int = 0,
        border_object: str = "wall",
    ):
        # Pass border information to the base class.
        super().__init__(border_width=border_width, border_object=border_object)
        self._width = width
        self._height = height
        self._radial_params = radial_params

        # Number of arms must be between 4 and 12.
        self._arms = radial_params.get("arms", 4)
        assert 4 <= self._arms <= 12, "Number of arms must be between 4 and 12"

        # Arm length: default to half of the smaller grid dimension minus one.
        self._arm_length = radial_params.get("arm_length", min(width, height) // 2 - 1)
        # Arm width: how many cells wide the corridor should be.
        self._arm_width = radial_params.get("arm_width", 4)

        self._rng = np.random.default_rng(seed)

        # Initialize grid filled with "wall" strings.
        self._grid = np.full((self._height, self._width), "wall", dtype='<U50')
        self._path_positions: Set[Tuple[int, int]] = set()

    def _build(self) -> np.ndarray:
        """
        Build the radial maze.
        
        1. Carve out arms (corridors) from the center using a Bresenham line,
           widening each arm to the specified arm width.
        2. At the end of arms 0, 1, and 2 place a generator, converter, and heart altar respectively.
        3. Place the agent at the center of the maze.
        """
        center_x = self._width // 2
        center_y = self._height // 2

        # Map special objects to arm indices.
        special_objects = {0: "generator", 1: "converter", 2: "altar"}

        # Carve each arm from the center.
        for arm_index in range(self._arms):
            angle = 2 * math.pi * arm_index / self._arms
            end_x = center_x + int(round(self._arm_length * math.cos(angle)))
            end_y = center_y + int(round(self._arm_length * math.sin(angle)))

            # Get the points along the central line of the arm.
            points: List[Tuple[int, int]] = self._bresenham_line(center_x, center_y, end_x, end_y)
            # Determine the offset range so that the corridor is centered.
            # For even widths, e.g. 4, we want offsets: -2, -1, 0, 1.
            if self._arm_width % 2 == 0:
                offset_range = range(-self._arm_width // 2, self._arm_width // 2)
            else:
                offset_range = range(-self._arm_width // 2, self._arm_width // 2 + 1)

            # For each point along the central line, carve out a block of cells.
            for (x, y) in points:
                for dx in offset_range:
                    for dy in offset_range:
                        new_x = x + dx
                        new_y = y + dy
                        if 0 <= new_x < self._width and 0 <= new_y < self._height:
                            self._grid[new_y, new_x] = "empty"
                            self._path_positions.add((new_x, new_y))

            # If this arm is designated for a special object, place it at the endpoint.
            if arm_index in special_objects:
                obj = special_objects[arm_index]
                if 0 <= end_x < self._width and 0 <= end_y < self._height:
                    self._grid[end_y, end_x] = obj

        # Place the agent in the center.
        self._grid[center_y, center_x] = "agent.agent"

        return self._grid

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """
        Generate the list of grid points from (x0, y0) to (x1, y1) using Bresenham's algorithm.
        """
        points: List[Tuple[int, int]] = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points.append((x1, y1))
        return points