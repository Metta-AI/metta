import math
from typing import Dict, Literal

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
    # distance metric for r: euclidean, manhattan, or traversal (BFS through empties)
    distance_metric: Literal["euclidean", "manhattan", "traversal"] = "euclidean"
    min_radius: int | None = None  # exclude inner radius (cells) if set
    clearance: int = 1  # empty ring around placed object
    carve: bool = True  # if True, carve exactly the 8 tiles around each placed object
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
            return bool(np.all(grid[y0:y1, x0:x1] == "empty"))

        # Precompute candidate cells and radial weights
        if self.params.carve:
            ys, xs = np.indices((height, width))
            empties = np.column_stack((ys.ravel(), xs.ravel()))
        else:
            empties = np.argwhere(grid == "empty")
        if empties.size == 0:
            return

        # Compute distances according to metric
        metric = self.params.distance_metric
        if metric == "manhattan":
            rs = np.abs(empties[:, 1] - cx) + np.abs(empties[:, 0] - cy)
        elif metric == "traversal":
            # BFS from center over cells considered passable ("empty")
            # Build mask of passable cells (treat empty as passable; others blocked)
            passable = (grid == "empty").astype(np.uint8)
            # Allow starting at center even if non-empty by treating it as passable temporarily
            passable[min(max(cy, 0), height - 1), min(max(cx, 0), width - 1)] = 1

            dist = np.full((height, width), np.inf, dtype=float)
            from collections import deque

            dq = deque()
            start_y, start_x = int(cy), int(cx)
            dist[start_y, start_x] = 0.0
            dq.append((start_y, start_x))
            # 4-neighborhood traversal distance
            for_y = (-1, 1, 0, 0)
            for_x = (0, 0, -1, 1)
            while dq:
                y, x = dq.popleft()
                base = dist[y, x]
                for dy, dx in zip(for_y, for_x, strict=True):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and passable[ny, nx] == 1 and dist[ny, nx] == np.inf:
                        dist[ny, nx] = base + 1.0
                        dq.append((ny, nx))
            rs = dist[empties[:, 0], empties[:, 1]]
            # For unreachable cells (inf), fall back to large finite value (treat as far)
            unreachable = ~np.isfinite(rs)
            if np.any(unreachable):
                finite_vals = rs[np.isfinite(rs)]
                fallback = (finite_vals.max() + 1.0) if finite_vals.size else float(height + width)
                rs[unreachable] = fallback
        else:
            rs = np.sqrt((empties[:, 1] - cx) ** 2 + (empties[:, 0] - cy) ** 2)
        if self.params.min_radius is not None:
            mask = rs >= self.params.min_radius
            empties = empties[mask]
            rs = rs[mask]
            if empties.size == 0:
                return

        # Avoid zero division
        # Normalize distance by rmax appropriate for metric
        if metric == "manhattan":
            rmax_metric = max(cx, width - 1 - cx) + max(cy, height - 1 - cy)
        else:
            rmax_metric = rmax
        rr = np.clip(rs / max(rmax_metric, 1e-6), 0.0, 1.0)
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
                if self.params.carve and not ok_with_clearance(x, y, 1):
                    # pre-clear a 3x3 neighborhood to ensure placement
                    x0 = max(0, x - 1)
                    x1 = min(width, x + 2)
                    y0 = max(0, y - 1)
                    y1 = min(height, y + 2)
                    grid[y0:y1, x0:x1] = "empty"

                if ok_with_clearance(x, y, self.params.clearance):
                    grid[y, x] = name
                    # ensure the 8 neighbors are empty (exact 3x3 ring)
                    if self.params.carve:
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                if dx == 0 and dy == 0:
                                    continue
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < height and 0 <= nx < width:
                                    grid[ny, nx] = "empty"
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
