from typing import Tuple

import numpy as np

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene


class BiomeCityParams(Config):
    # Grid of city blocks; roads form the gaps between blocks
    pitch: int = 10  # distance between block starts
    road_width: int = 2  # empty corridors
    place_prob: float = 0.9  # probability to place a block in a cell
    min_block_frac: float = 0.6  # fraction of pitch used for block (before jitter)
    jitter: int = 2  # random size jitter applied to block width/height
    dither_edges: bool = True  # Add organic edge noise
    dither_prob: float = 0.15  # Probability to flip edge cells
    dither_depth: int = 5  # How many cells deep to consider as edge zone


class BiomeCity(Scene[BiomeCityParams]):
    """
    City-like layout made of rectangular blocks separated by roads.

    Produces a rectilinear grid of roads (empty) and fills blocks with walls.
    """

    def _clip_rect(self, x0: int, y0: int, w: int, h: int) -> Tuple[int, int, int, int]:
        x0 = max(0, x0)
        y0 = max(0, y0)
        w = max(0, min(w, self.width - x0))
        h = max(0, min(h, self.height - y0))
        return x0, y0, w, h

    def render(self):
        grid = self.grid
        H, W = self.height, self.width
        p = self.params

        pitch = max(4, int(p.pitch))
        road_w = max(1, int(p.road_width))
        min_block = max(1, int(pitch * float(p.min_block_frac)))
        jitter = max(0, int(p.jitter))

        # Ensure we have empty background to carve roads into; preserve any prior walls
        # We will only add walls for blocks and leave existing empties/roads.

        # Iterate over block grid cells
        for gy in range(0, H, pitch):
            for gx in range(0, W, pitch):
                if self.rng.random() > p.place_prob:
                    continue

                # Compute block rectangle leaving road margins on all sides
                x0 = gx + road_w
                y0 = gy + road_w
                bw = min_block + int(self.rng.integers(-jitter, jitter + 1))
                bh = min_block + int(self.rng.integers(-jitter, jitter + 1))

                # Avoid overlapping into next road
                bw = min(bw, pitch - 2 * road_w)
                bh = min(bh, pitch - 2 * road_w)

                if bw <= 0 or bh <= 0:
                    continue

                x0, y0, bw, bh = self._clip_rect(x0, y0, bw, bh)
                if bw == 0 or bh == 0:
                    continue

                grid[y0 : y0 + bh, x0 : x0 + bw] = "wall"

        # To emphasize roads, optionally thicken the road network by clearing stripes
        # along the grid every pitch cells
        # Horizontal roads
        for gy in range(0, H, pitch):
            y0 = gy
            y1 = min(H, gy + road_w)
            grid[y0:y1, :] = np.where(grid[y0:y1, :] == "wall", "wall", "empty")
        # Vertical roads
        for gx in range(0, W, pitch):
            x0 = gx
            x1 = min(W, gx + road_w)
            grid[:, x0:x1] = np.where(grid[:, x0:x1] == "wall", "wall", "empty")

        # Apply edge dithering for organic look
        if p.dither_edges:
            self._dither_edges(grid, p.dither_prob)

    def _dither_edges(self, grid, prob: float):
        """Add organic noise to edges between wall and empty cells."""
        H, W = grid.shape
        depth = self.params.dither_depth

        # Find edges: cells within 'depth' distance of opposite type
        for y in range(depth, H - depth):
            for x in range(depth, W - depth):
                current = grid[y, x]

                # Find distance to nearest opposite type cell
                min_dist = depth + 1
                for dy in range(-depth, depth + 1):
                    for dx in range(-depth, depth + 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            if grid[ny, nx] != current:
                                dist = max(abs(dy), abs(dx))  # Chebyshev distance
                                min_dist = min(min_dist, dist)

                # Only dither cells near edges (within depth)
                if min_dist <= depth:
                    # Probability increases as we get closer to edge
                    # At distance 1: prob, at distance depth: prob/depth
                    edge_prob = prob * (depth - min_dist + 1) / depth
                    if self.rng.random() < edge_prob:
                        grid[y, x] = "empty" if current == "wall" else "wall"
