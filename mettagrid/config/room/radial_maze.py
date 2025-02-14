from typing import Set, Tuple, Union, List
import numpy as np
import math
from omegaconf import DictConfig

from mettagrid.config.room.room import Room


class RadialMaze(Room):
    """
    A generalizable radial maze map with a central starting position.
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
        super().__init__(border_width=border_width, border_object=border_object)
        self._width, self._height = width, height
        self._radial_params = radial_params
        self._arms = radial_params.get("arms", 4)
        assert 4 <= self._arms <= 12, "Number of arms must be between 4 and 12"
        self._arm_length = radial_params.get("arm_length", min(width, height) // 2 - 1)
        self._arm_width = radial_params.get("arm_width", 4)
        self._rng = np.random.default_rng(seed)
        self._grid = np.full((height, width), "wall", dtype='<U50')
        self._path_positions: Set[Tuple[int, int]] = set()

    def _build(self) -> np.ndarray:
        center_x, center_y = self._width // 2, self._height // 2
        special_objects = {0: "generator", 1: "converter", 2: "altar"}
        for arm in range(self._arms):
            angle = 2 * math.pi * arm / self._arms
            end_x = center_x + int(round(self._arm_length * math.cos(angle)))
            end_y = center_y + int(round(self._arm_length * math.sin(angle)))
            points: List[Tuple[int, int]] = self._bresenham_line(center_x, center_y, end_x, end_y)
            offset_range = range(-self._arm_width // 2, self._arm_width // 2 + (self._arm_width % 2))
            for x, y in points:
                for dx in offset_range:
                    for dy in offset_range:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self._width and 0 <= ny < self._height:
                            self._grid[ny, nx] = "empty"
                            self._path_positions.add((nx, ny))
            if arm in special_objects and 0 <= end_x < self._width and 0 <= end_y < self._height:
                self._grid[end_y, end_x] = special_objects[arm]
        self._grid[center_y, center_x] = "agent.agent"
        return self._grid

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        points: List[Tuple[int, int]] = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        x, y = x0, y0
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
