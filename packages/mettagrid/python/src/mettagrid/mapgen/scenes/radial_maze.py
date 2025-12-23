import math

import numpy as np
from pydantic import Field

from mettagrid.mapgen.scene import Scene, SceneConfig
from mettagrid.mapgen.utils.draw import bresenham_line


class RadialMazeConfig(SceneConfig):
    arms: int = Field(default=4, ge=4, le=12)
    arm_width: int = Field(default=4, ge=1)
    arm_length: int | None = None
    clear_background: bool = Field(default=True, description="If True, fill area with walls before carving arms")
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

        offsets = np.arange(-arm_width // 2, arm_width // 2 + (arm_width % 2))

        endpoints: list[tuple[int, int]] = []
        for arm in range(self.config.arms):
            angle = 2 * math.pi * arm / self.config.arms
            ex = cx + int(round(arm_length * math.cos(angle)))
            ey = cy + int(round(arm_length * math.sin(angle)))
            pts = np.array(bresenham_line(cx, cy, ex, ey), dtype=int)
            for dx in offsets:
                for dy in offsets:
                    xs = pts[:, 0] + dx
                    ys = pts[:, 1] + dy
                    mask = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
                    xs_in = xs[mask]
                    ys_in = ys[mask]
                    carved[ys_in, xs_in] = True
                    self.grid[ys_in, xs_in] = "empty"

            # Last in-bounds point on the ray
            for px, py in pts[::-1]:
                if 0 <= px < self.width and 0 <= py < self.height:
                    endpoints.append((px, py))
                    break

        if self.config.outline_walls:
            # Cells adjacent (8-neighbor) to carved but not carved themselves.
            def _shift(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
                out = np.zeros_like(mask, dtype=bool)
                y_src_start = max(0, -dy)
                y_src_end = mask.shape[0] - max(0, dy)
                x_src_start = max(0, -dx)
                x_src_end = mask.shape[1] - max(0, dx)

                y_dst_start = max(0, dy)
                y_dst_end = y_dst_start + (y_src_end - y_src_start)
                x_dst_start = max(0, dx)
                x_dst_end = x_dst_start + (x_src_end - x_src_start)

                if y_src_end > y_src_start and x_src_end > x_src_start:
                    out[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = mask[
                        y_src_start:y_src_end, x_src_start:x_src_end
                    ]
                return out

            neighbor_any = np.zeros_like(carved, dtype=bool)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor_any |= _shift(carved, dx, dy)

            outline = neighbor_any & (~carved)
            self.grid[outline] = "wall"

        for x_end, y_end in endpoints:
            self.make_area(x_end, y_end, 1, 1, tags=["endpoint"])

        # this could be found with Layout, but having a designated area is more convenient
        self.make_area(cx, cy, 1, 1, tags=["center"])
