from typing import Literal

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene

Axis = Literal["horizontal", "vertical"]


class EnforceSymmetryParams(Config):
    horizontal: bool = False
    vertical: bool = False


class EnforceSymmetry(Scene[EnforceSymmetryParams]):
    """
    In-place symmetry enforcement on the current grid.

    - horizontal=True: mirror top half onto bottom
    - vertical=True: mirror left half onto right
    Apply both for 4-way symmetry.
    """

    def render(self):
        h, w = self.height, self.width

        if self.params.vertical:
            mid = w // 2
            # copy left -> right, leave center column if odd
            for x in range(mid):
                x2 = w - 1 - x
                self.grid[:, x2] = self.grid[:, x]

        if self.params.horizontal:
            mid = h // 2
            # copy top -> bottom, leave center row if odd
            for y in range(mid):
                y2 = h - 1 - y
                self.grid[y2, :] = self.grid[y, :]
