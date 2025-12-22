import numpy as np

from mettagrid.mapgen.scene import Scene, SceneConfig
from mettagrid.mapgen.scenes.dither import dither_edges


class AsteroidMaskConfig(SceneConfig):
    radius_scale: float = 1.0
    shape_power: float = 8.0
    jagged_prob: float = 0.35
    jagged_depth: int = 5


class AsteroidMask(Scene[AsteroidMaskConfig]):
    def render(self) -> None:
        grid = self.grid
        height, width = self.height, self.width
        cfg = self.config

        radius_scale = max(0.1, float(cfg.radius_scale))
        power = max(1.0, float(cfg.shape_power))
        rx = max(1.0, (width - 1) * 0.5 * radius_scale)
        ry = max(1.0, (height - 1) * 0.5 * radius_scale)

        ys, xs = np.ogrid[:height, :width]
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5
        nx = np.abs(xs - cx) / rx
        ny = np.abs(ys - cy) / ry
        superellipse = (nx**power) + (ny**power)

        mask_grid = np.where(superellipse <= 1.0, "empty", "wall")
        if cfg.jagged_depth > 0 and cfg.jagged_prob > 0:
            depth = int(cfg.jagged_depth)
            prob = float(cfg.jagged_prob)

            wall = mask_grid == "wall"
            empty = ~wall

            def _expand(mask: np.ndarray) -> np.ndarray:
                up = np.zeros_like(mask, dtype=bool)
                down = np.zeros_like(mask, dtype=bool)
                left = np.zeros_like(mask, dtype=bool)
                right = np.zeros_like(mask, dtype=bool)
                up[:-1] = mask[1:]
                down[1:] = mask[:-1]
                left[:, :-1] = mask[:, 1:]
                right[:, 1:] = mask[:, :-1]
                up_left = np.zeros_like(mask, dtype=bool)
                up_right = np.zeros_like(mask, dtype=bool)
                down_left = np.zeros_like(mask, dtype=bool)
                down_right = np.zeros_like(mask, dtype=bool)
                up_left[:-1, :-1] = mask[1:, 1:]
                up_right[:-1, 1:] = mask[1:, :-1]
                down_left[1:, :-1] = mask[:-1, 1:]
                down_right[1:, 1:] = mask[:-1, :-1]
                return up | down | left | right | up_left | up_right | down_left | down_right

            boundary_inside = _expand(wall) & empty
            dist = np.full(mask_grid.shape, np.inf, dtype=np.float32)
            dist[boundary_inside] = 1.0
            seen = boundary_inside.copy()
            frontier = boundary_inside
            current_depth = 1

            while current_depth < depth and frontier.any():
                frontier = _expand(frontier) & empty & (~seen)
                if not frontier.any():
                    break
                current_depth += 1
                dist[frontier] = current_depth
                seen |= frontier

            reachable = dist <= depth
            if reachable.any():
                edge_prob = prob * (depth - dist + 1) / depth
                flips = (self.rng.random(mask_grid.shape) < edge_prob) & reachable
                mask_grid[flips] = "wall"

        grid[mask_grid == "wall"] = "wall"
