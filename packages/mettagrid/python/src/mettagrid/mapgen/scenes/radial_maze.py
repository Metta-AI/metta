import math

import numpy as np
from pydantic import Field

from mettagrid.mapgen.scene import Scene, SceneConfig
from mettagrid.mapgen.utils.draw import bresenham_line


class RadialMazeConfig(SceneConfig):
    arms: int = Field(default=4, ge=4, le=12)
    arm_width: int = Field(default=4, ge=1)
    arm_length: int | None = None
    clear_background: bool = Field(default=False, description="If True, fill area with walls before carving arms")
    outline_walls: bool = Field(default=True, description="Outline arms with walls for visual clarity")


class RadialMaze(Scene[RadialMazeConfig]):
    """A radial maze with a central starting position."""

    def render(self):
        arm_length = self.config.arm_length or min(self.width, self.height) // 2 - 1
        arm_width = self.config.arm_width
        if self.config.clear_background:
            self.grid[:] = "wall"

        cx, cy = self.width // 2, self.height // 2

        carved = np.zeros((self.height, self.width), dtype=bool)

        for arm in range(self.config.arms):
            angle = 2 * math.pi * arm / self.config.arms
            ex = cx + int(round(arm_length * math.cos(angle)))
            ey = cy + int(round(arm_length * math.sin(angle)))
            points = bresenham_line(cx, cy, ex, ey)
            offsets = range(-arm_width // 2, arm_width // 2 + (arm_width % 2))
            for x, y in points:
                for dx in offsets:
                    for dy in offsets:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            self.grid[ny, nx] = "empty"
                            carved[ny, nx] = True

            # Choose the last in-bound point from the arm's path.
            special_point = None
            for p in reversed(points):
                px, py = p
                if 0 <= px < self.width and 0 <= py < self.height:
                    special_point = p
                    break
            if special_point is not None:
                self.make_area(special_point[0], special_point[1], 1, 1, tags=["endpoint"])

        if self.config.outline_walls:
            outline = np.zeros_like(carved)
            H, W = carved.shape
            for y in range(H):
                for x in range(W):
                    if not carved[y, x]:
                        continue
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W and not carved[ny, nx]:
                                outline[ny, nx] = True
            # Draw walls on outline cells that are not part of the carved corridors
            mask = outline & (~carved)
            self.grid[mask] = "wall"

        # this could be found with Layout, but having a designated area is more convenient
        self.make_area(cx, cy, 1, 1, tags=["center"])
