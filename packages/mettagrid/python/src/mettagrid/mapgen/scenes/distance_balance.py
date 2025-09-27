import logging
from typing import Dict, List, Tuple

import numpy as np

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene

logger = logging.getLogger(__name__)


class DistanceBalanceParams(Config):
    converter_types: List[str] = ["generator_red", "generator_blue", "generator_green", "lab"]
    tolerance: float = 8.0  # allowed deviation from global mean
    balance: bool = False  # if True, carve shortcuts for far-away types
    carves_per_type: int = 2
    carve_width: int = 1
    relocate: bool = True  # if True, move far-away converters toward target mean
    moves_per_type: int = 2
    relocation_clearance: int = 1
    relocation_min_radius: int = 6  # don't place too close to altar


class DistanceBalance(Scene[DistanceBalanceParams]):
    """
    Compute distances from the central altar to each converter type and optionally carve shortcuts
    to roughly equalize mean distances across types.
    """

    def _find_center(self) -> Tuple[int, int] | None:
        ys, xs = np.where(self.grid == "altar")
        if ys.size == 0:
            return None
        # take the first (should be unique)
        return int(xs[0]), int(ys[0])

    def _bfs_distances(self, passable_values: Tuple[str, ...] = ("empty",)) -> np.ndarray:
        h, w = self.height, self.width
        dist = np.full((h, w), np.inf)
        start = self._find_center()
        if start is None:
            return dist
        sx, sy = start
        from collections import deque

        dq = deque()
        if self.grid[sy, sx] in passable_values or self.grid[sy, sx] == "altar":
            dist[sy, sx] = 0
            dq.append((sy, sx))

        while dq:
            y, x = dq.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if (self.grid[ny, nx] in passable_values) and dist[ny, nx] == np.inf:
                        dist[ny, nx] = dist[y, x] + 1
                        dq.append((ny, nx))
        return dist

    def _carve_line(self, x0: int, y0: int, x1: int, y1: int, width: int = 1):
        # Bresenham-like line carving; only turns walls into empty
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        h, w = self.height, self.width
        while True:
            x0b = max(0, x - width // 2)
            x1b = min(w, x + width // 2 + 1)
            y0b = max(0, y - width // 2)
            y1b = min(h, y + width // 2 + 1)
            sub = self.grid[y0b:y1b, x0b:x1b]
            sub[sub == "wall"] = "empty"
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def render(self):
        center = self._find_center()
        if center is None:
            logger.info("DistanceBalance: no altar found; skipping")
            return
        cx, cy = center

        # Compute distances on current passable cells
        dist = self._bfs_distances()

        # Collect positions per converter type
        positions: Dict[str, list[Tuple[int, int]]] = {t: [] for t in self.params.converter_types}
        for t in self.params.converter_types:
            ys, xs = np.where(self.grid == t)
            for y, x in zip(ys, xs, strict=False):
                positions[t].append((int(x), int(y)))

        type_means: Dict[str, float] = {}
        all_dists: list[float] = []
        for t, pts in positions.items():
            ds: list[float] = []
            for x, y in pts:
                # Converter cells are not passable; measure via nearest adjacent passable cell
                neighbor_ds: list[float] = []
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        d = dist[ny, nx]
                        if np.isfinite(d):
                            neighbor_ds.append(float(d) + 1.0)
                if neighbor_ds:
                    ds.append(float(min(neighbor_ds)))
            if ds:
                m = float(np.mean(ds))
                type_means[t] = m
                all_dists.extend(ds)
            else:
                type_means[t] = float("inf")

        if all_dists:
            global_mean = float(np.mean(all_dists))
        else:
            global_mean = float("inf")

        logger.info("Converter distance means (global mean %.2f): %s", global_mean, type_means)

        # Relocate far-out types to cells closer to the global mean
        if np.isfinite(global_mean) and self.params.relocate:
            passable = np.isfinite(dist)
            cand_ys, cand_xs = np.where(passable)
            cand_ds = dist[cand_ys, cand_xs]

            def ok_clearance(x: int, y: int, c: int) -> bool:
                x0 = max(0, x - c)
                x1 = min(self.width, x + c + 1)
                y0 = max(0, y - c)
                y1 = min(self.height, y + c + 1)
                sub = self.grid[y0:y1, x0:x1]
                return np.all(sub == "empty")

            for t, mean_d in type_means.items():
                if mean_d > global_mean + self.params.tolerance:
                    pts = positions[t]
                    pts_sorted = sorted(
                        pts,
                        key=lambda p: dist[p[1], p[0]] if np.isfinite(dist[p[1], p[0]]) else 1e9,
                        reverse=True,
                    )
                    to_move = pts_sorted[: self.params.moves_per_type]

                    idx_pool = np.where(
                        (cand_ds >= self.params.relocation_min_radius) & (np.abs(cand_ds - global_mean) <= 2.0)
                    )[0]
                    choice_order = self.rng.permutation(idx_pool) if idx_pool.size else np.array([], dtype=int)

                    for ox, oy in to_move:
                        placed = False
                        for k in choice_order:
                            ny, nx = int(cand_ys[k]), int(cand_xs[k])
                            if ok_clearance(nx, ny, self.params.relocation_clearance):
                                self.grid[oy, ox] = "empty"
                                if self.params.relocation_clearance > 0:
                                    x0 = max(0, nx - self.params.relocation_clearance)
                                    x1 = min(self.width, nx + self.params.relocation_clearance + 1)
                                    y0 = max(0, ny - self.params.relocation_clearance)
                                    y1 = min(self.height, ny + self.params.relocation_clearance + 1)
                                    self.grid[y0:y1, x0:x1] = "empty"
                                self.grid[ny, nx] = t
                                placed = True
                                break
                        if not placed:
                            logger.debug("DistanceBalance: relocation slot not found for %s", t)

        # Optional carving after relocation to ensure connectivity
        if self.params.balance and np.isfinite(global_mean):
            for t, mean_d in type_means.items():
                if not np.isfinite(mean_d):
                    to_fix = positions[t][: self.params.carves_per_type]
                elif mean_d > global_mean + self.params.tolerance:
                    pts = positions[t]
                    pts_sorted = sorted(
                        pts,
                        key=lambda p: dist[p[1], p[0]] if np.isfinite(dist[p[1], p[0]]) else 1e9,
                        reverse=True,
                    )
                    to_fix = pts_sorted[: self.params.carves_per_type]
                else:
                    to_fix = []

                for x, y in to_fix:
                    self._carve_line(x, y, cx, cy, width=self.params.carve_width)
