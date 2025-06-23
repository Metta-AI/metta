import math
from typing import Set, Tuple, Union

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room
from metta.mettagrid.room.utils import bresenham_line, create_grid


class RadialMaze(Room):
    """A radial maze with a central starting position."""

    def __init__(
        self,
        width: int,
        height: int,
        radial_params: DictConfig,
        seed: Union[int, None] = None,
        border_width: int = 1,
        border_object: str = "wall",
        onlyhearts: bool = False,
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._width, self._height = width, height
        self._radial_params = radial_params
        self._arms = radial_params.get("arms", 4)
        assert 4 <= self._arms <= 12, "Number of arms must be between 4 and 12"
        self._arm_length = radial_params.get("arm_length", min(width, height) // 2 - 1)
        self._arm_width = radial_params.get("arm_width", 4)
        self._rng = np.random.default_rng(seed)
        self._onlyhearts = onlyhearts

    def _build(self) -> np.ndarray:
        grid = create_grid(self._height, self._width, fill_value="wall")
        path_positions: Set[Tuple[int, int]] = set()

        cx, cy = self._width // 2, self._height // 2

        if self._onlyhearts:
            specials = {0: "altar", 1: "altar", 2: "altar"}
        else:
            specials = {0: "mine", 1: "generator", 2: "altar"}

        special_endpoints = {}

        for arm in range(self._arms):
            angle = 2 * math.pi * arm / self._arms
            ex = cx + int(round(self._arm_length * math.cos(angle)))
            ey = cy + int(round(self._arm_length * math.sin(angle)))
            points = bresenham_line(cx, cy, ex, ey)
            offsets = range(-self._arm_width // 2, self._arm_width // 2 + (self._arm_width % 2))
            for x, y in points:
                for dx in offsets:
                    for dy in offsets:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self._width and 0 <= ny < self._height:
                            grid[ny, nx] = "empty"
                            path_positions.add((nx, ny))
            if arm in specials:
                # Choose the last in-bound point from the arm's path.
                special_point = None
                for p in reversed(points):
                    px, py = p
                    if 0 <= px < self._width and 0 <= py < self._height:
                        special_point = p
                        break
                if special_point is not None:
                    special_endpoints[arm] = special_point

        for arm, label in specials.items():
            if arm in special_endpoints:
                ex, ey = special_endpoints[arm]
                grid[ey, ex] = label

        grid[cy, cx] = "agent.agent"
        return grid
