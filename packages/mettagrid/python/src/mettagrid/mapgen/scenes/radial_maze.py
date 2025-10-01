import math

from pydantic import Field

from mettagrid.mapgen.scene import Scene, SceneConfig
from mettagrid.mapgen.utils.draw import bresenham_line


class RadialMazeConfig(SceneConfig):
    arms: int = Field(default=4, ge=4, le=12)
    arm_width: int = Field(default=4, ge=1)
    arm_length: int | None = None


class RadialMaze(Scene[RadialMazeConfig]):
    """A radial maze with a central starting position."""

    def render(self):
        arm_length = self.config.arm_length or min(self.width, self.height) // 2 - 1
        arm_width = self.config.arm_width
        self.grid[:] = "wall"

        cx, cy = self.width // 2, self.height // 2

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

            # Choose the last in-bound point from the arm's path.
            special_point = None
            for p in reversed(points):
                px, py = p
                if 0 <= px < self.width and 0 <= py < self.height:
                    special_point = p
                    break
            if special_point is not None:
                self.make_area(special_point[0], special_point[1], 1, 1, tags=["endpoint"])

        # this could be found with Layout, but having a designated area is more convenient
        self.make_area(cx, cy, 1, 1, tags=["center"])
