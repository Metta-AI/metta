import math

from pydantic import Field

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene
from mettagrid.mapgen.utils.draw import bresenham_line


class RadialMazeParams(Config):
    arms: int = Field(default=4, ge=4, le=12)
    arm_width: int = Field(default=4, ge=1)
    arm_length: int | None = None
    fill_background: bool = Field(default=True)  # If False, only draws the arms without filling the grid


class RadialMaze(Scene[RadialMazeParams]):
    """A radial maze with a central starting position."""

    def render(self):
        arm_length = self.params.arm_length or min(self.width, self.height) // 2 - 1
        arm_width = self.params.arm_width

        # Fill background with walls
        if self.params.fill_background:
            self.grid[:] = "wall"
        else:
            # When not filling background, draw circular wall pattern instead
            # Draw walls between the arms (spoke walls)
            cx, cy = self.width // 2, self.height // 2
            for y in range(self.height):
                for x in range(self.width):
                    dx, dy = x - cx, y - cy
                    distance = math.sqrt(dx * dx + dy * dy)
                    if distance > 1 and distance < arm_length:
                        # Calculate angle for this point
                        angle = math.atan2(dy, dx)
                        if angle < 0:
                            angle += 2 * math.pi

                        # Check if this point is NOT in any arm
                        in_arm = False
                        for arm_idx in range(self.params.arms):
                            arm_angle = 2 * math.pi * arm_idx / self.params.arms
                            angle_diff = abs(angle - arm_angle)
                            # Normalize angle difference to [-pi, pi]
                            if angle_diff > math.pi:
                                angle_diff = 2 * math.pi - angle_diff

                            # Calculate angular width based on arm_width
                            angular_width = math.atan2(arm_width / 2, distance)
                            if angle_diff < angular_width:
                                in_arm = True
                                break

                        # If not in any arm, draw a wall
                        if not in_arm:
                            self.grid[y, x] = "wall"

        cx, cy = self.width // 2, self.height // 2

        # Draw the arms (always carved as empty regardless of fill_background)
        for arm in range(self.params.arms):
            angle = 2 * math.pi * arm / self.params.arms
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
