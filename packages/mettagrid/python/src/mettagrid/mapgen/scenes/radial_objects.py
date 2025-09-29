import math
from typing import Dict

import numpy as np

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene


class RadialObjectsParams(Config):
    objects: Dict[str, int] = {}
    # mode controls the radial weighting function
    mode: str = "power"  # one of: power, exp, log, gaussian
    k: float = 3.0  # for power: (r/rmax)^k
    alpha: float = 5.0  # for exp: exp(alpha * (r/rmax - 1))
    beta: float = 0.1  # for log: log(1 + beta * r)
    mu: float = 0.75  # for gaussian: center at mu*rmax
    sigma: float = 0.1  # for gaussian: std fraction of rmax
    min_radius: int | None = None  # exclude inner radius (cells) if set
    clearance: int = 1  # empty ring around placed object
    carve: bool = False  # if True, carve clearance area to empty before placing
    max_trials_per_object: int = 5000


class RadialObjects(Scene[RadialObjectsParams]):
    """
    Place given objects with probability increasing with distance from center.

    - Bias controlled by `k`: p ~ (r / rmax)^k
    - Optional `min_radius` excludes placements too near the center
    - Respects a 1-cell clearance by default
    """

    def render(self):
        grid = self.grid
        height, width = grid.shape

        cx, cy = width // 2, height // 2
        rmax = math.hypot(max(cx, width - 1 - cx), max(cy, height - 1 - cy))

        def ok_with_clearance(x: int, y: int, clearance: int) -> bool:
            if not (0 <= x < width and 0 <= y < height):
                return False
            x0 = max(0, x - clearance)
            x1 = min(width, x + clearance + 1)
            y0 = max(0, y - clearance)
            y1 = min(height, y + clearance + 1)
            return np.all(grid[y0:y1, x0:x1] == "empty")

        # Precompute candidate cells and radial weights
        if self.params.carve:
            ys, xs = np.indices((height, width))
            empties = np.column_stack((ys.ravel(), xs.ravel()))
        else:
            empties = np.argwhere(grid == "empty")
        if empties.size == 0:
            return

        rs = np.sqrt((empties[:, 1] - cx) ** 2 + (empties[:, 0] - cy) ** 2)
        if self.params.min_radius is not None:
            mask = rs >= self.params.min_radius
            empties = empties[mask]
            rs = rs[mask]
            if empties.size == 0:
                return

        # Avoid zero division
        rr = np.clip(rs / max(rmax, 1e-6), 0.0, 1.0)
        mode = self.params.mode
        if mode == "power":
            ws = np.power(rr, self.params.k)
        elif mode == "exp":
            # favor edges as alpha increases
            ws = np.exp(self.params.alpha * (rr - 1.0))
        elif mode == "log":
            # gentle increase; ensure positive
            ws = np.log1p(self.params.beta * rs)
        elif mode == "gaussian":
            # peak around mu*rmax
            mu = self.params.mu
            sigma = max(self.params.sigma, 1e-6)
            ws = np.exp(-0.5 * ((rr - mu) / sigma) ** 2)
        else:
            ws = np.power(rr, self.params.k)
        ws = ws + 1e-9
        ws = ws / ws.sum()

        for name, count in self.params.objects.items():
            placed = 0
            attempts = 0
            # We'll sample by weight; after successful placement, remove the cleared neighborhood from candidates
            while placed < count and attempts < self.params.max_trials_per_object and len(ws) > 0:
                attempts += 1
                idx = int(self.rng.choice(len(ws), p=ws))
                y, x = int(empties[idx][0]), int(empties[idx][1])
                # carve if requested
                if self.params.carve and not ok_with_clearance(x, y, self.params.clearance):
                    x0 = max(0, x - self.params.clearance)
                    x1 = min(width, x + self.params.clearance + 1)
                    y0 = max(0, y - self.params.clearance)
                    y1 = min(height, y + self.params.clearance + 1)
                    grid[y0:y1, x0:x1] = "empty"

                if ok_with_clearance(x, y, self.params.clearance):
                    grid[y, x] = name
                    placed += 1
                    # remove neighbors within clearance from candidate set
                    keep = []
                    for i, (yy, xx) in enumerate(empties):
                        if abs(int(xx) - x) > self.params.clearance or abs(int(yy) - y) > self.params.clearance:
                            keep.append(i)
                    empties = empties[keep]
                    if len(keep) > 0:
                        ws = ws[keep]
                        ws = ws / ws.sum()
                    else:
                        ws = np.array([])
                else:
                    # remove this candidate to avoid re-trying same blocked cell
                    if len(ws) > 1:
                        empties = np.delete(empties, idx, axis=0)
                        ws = np.delete(ws, idx)
                        ws = ws / ws.sum()
                    else:
                        break
