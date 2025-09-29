import numpy as np

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene


class BiomeDesertParams(Config):
    dune_period: int = 8  # distance between dune ridges
    ridge_width: int = 2  # ridge thickness in cells
    angle: float = 0.0  # radians; 0 == vertical stripes, pi/4 diagonal, etc.
    noise_prob: float = 0.3  # pepper dunes with small gaps


class BiomeDesert(Scene[BiomeDesertParams]):
    """
    Striated dunes: parallel ridges with occasional gaps.
    Walls are dunes; empty is sand path.
    """

    def render(self):
        grid = self.grid
        H, W = self.height, self.width
        p = self.params

        period = max(2, int(p.dune_period))
        width = max(1, int(p.ridge_width))
        theta = float(p.angle)

        # Build coordinate field rotated by theta
        ys, xs = np.indices((H, W))
        # Rotate coordinates
        xr = xs * np.cos(theta) + ys * np.sin(theta)
        mask = (xr % period) < width

        dunes = mask.astype(np.uint8)
        # Pepper with noise holes, increase probability to 0.3
        holes = self.rng.random((H, W)) < p.noise_prob
        dunes = np.where(holes, 0, dunes)

        grid[dunes == 1] = "wall"
