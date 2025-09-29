from typing import List

import numpy as np

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene


class CarveConvertersParams(Config):
    converter_types: List[str] = [
        "generator_red",
        "generator_blue",
        "generator_green",
        "lab",
    ]
    clearance: int = 1  # radius for carving; 1 => 3x3 neighborhood


class CarveConverters(Scene[CarveConvertersParams]):
    """Final pass: clear a ring around every converter to prevent blockages.

    Clears a square neighborhood of (2*clearance+1)^2 centered on each converter.
    """

    def render(self):
        grid = self.grid
        h, w = self.height, self.width
        ys, xs = np.where(np.isin(grid, self.params.converter_types))
        r = max(0, int(self.params.clearance))
        for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
            orig = grid[y, x]
            y0 = max(0, y - r)
            y1 = min(h, y + r + 1)
            x0 = max(0, x - r)
            x1 = min(w, x + r + 1)
            grid[y0:y1, x0:x1] = "empty"
            grid[y, x] = orig  # restore center converter
