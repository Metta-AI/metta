from __future__ import annotations

import math
from typing import Optional

import numpy as np

from metta.mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from metta.mettagrid.map_builder.utils import draw_border


class ForagingMapBuilder(MapBuilder):
    class Config(MapBuilderConfig["ForagingMapBuilder"]):
        """
        Procedural hub-and-outside map builder for foraging tasks.

        Places a single agent in the center, a set of hub objects near the center,
        and a set of outside objects scattered at configurable radii.
        """

        seed: Optional[int] = None

        width: int = 13
        height: int = 13

        # Border configuration
        border_width: int = 0
        border_object: str = "wall"

        # Object placement
        # Map of object name -> count for hub (near center)
        hub_objects: dict[str, int] = {}
        # Map of object name -> count for outside (farther from center)
        outside_objects: dict[str, int] = {}

        # Hub layout and extent
        hub_layout: str = "grid"  # one of {"grid", "ring", "cross"}
        hub_box_radius: int = 1  # radius of the square box around center to place hub objects

        # Outside placement radii (Manhattan distance from center)
        outside_min_radius: int = 3
        outside_max_radius: int = 5

        # Whether to place the agent at the exact center
        center_agent: bool = True

        # Keep at least this Manhattan radius around the agent empty to avoid trapping
        agent_safe_radius: int = 1

    def __init__(self, config: Config):
        self._config = config
        self._rng = np.random.default_rng(self._config.seed)

    def _center(self) -> tuple[int, int]:
        return self._config.height // 2, self._config.width // 2

    def _in_bounds(self, r: int, c: int) -> bool:
        bw = self._config.border_width
        return bw <= r < self._config.height - bw and bw <= c < self._config.width - bw

    def _candidate_hub_positions(self, grid: np.ndarray) -> list[tuple[int, int]]:
        center_r, center_c = self._center()
        positions: list[tuple[int, int]] = []
        R = max(1, self._config.hub_box_radius)
        safe_r = max(0, self._config.agent_safe_radius)

        if self._config.hub_layout == "grid":
            for dr in range(-R, R + 1):
                for dc in range(-R, R + 1):
                    rr, cc = center_r + dr, center_c + dc
                    # avoid center and the safe ring around it
                    if (abs(dr) + abs(dc) <= safe_r) or not self._in_bounds(rr, cc):
                        continue
                    positions.append((rr, cc))
        elif self._config.hub_layout == "ring":
            # Ring at exact Manhattan radius R
            for dr in range(-R, R + 1):
                dc = R - abs(dr)
                for sdc in (-dc, dc):
                    rr, cc = center_r + dr, center_c + sdc
                    # avoid center and the safe ring
                    if (abs(dr) + abs(sdc) <= safe_r) or dc == 0 or not self._in_bounds(rr, cc):
                        continue
                    positions.append((rr, cc))
            for dc in range(-R, R + 1):
                dr = R - abs(dc)
                for sdr in (-dr, dr):
                    rr, cc = center_r + sdr, center_c + dc
                    if (abs(sdr) + abs(dc) <= safe_r) or dr == 0 or not self._in_bounds(rr, cc):
                        continue
                    positions.append((rr, cc))
            # Deduplicate
            positions = list(dict.fromkeys(positions))
        elif self._config.hub_layout == "cross":
            for d in range(1, R + 1):
                for rr, cc in (
                    (center_r + d, center_c),
                    (center_r - d, center_c),
                    (center_r, center_c + d),
                    (center_r, center_c - d),
                ):
                    if self._in_bounds(rr, cc) and d > safe_r:
                        positions.append((rr, cc))
        else:
            # Fallback to grid
            for dr in range(-R, R + 1):
                for dc in range(-R, R + 1):
                    rr, cc = center_r + dr, center_c + dc
                    if (abs(dr) + abs(dc) <= safe_r) or not self._in_bounds(rr, cc):
                        continue
                    positions.append((rr, cc))

        # Filter out already-occupied cells
        positions = [pos for pos in positions if grid[pos[0], pos[1]] == "empty"]
        return positions

    def _place_hub_objects(self, grid: np.ndarray) -> None:
        hub_symbols: list[str] = []
        for name, count in self._config.hub_objects.items():
            hub_symbols.extend([name] * count)

        if not hub_symbols:
            return

        positions = self._candidate_hub_positions(grid)
        # Prefer nearer positions first
        center_r, center_c = self._center()
        positions.sort(key=lambda rc: abs(rc[0] - center_r) + abs(rc[1] - center_c))

        i = 0
        for pos in positions:
            if i >= len(hub_symbols):
                break
            r, c = pos
            if grid[r, c] == "empty":
                grid[r, c] = hub_symbols[i]
                i += 1

    def _random_outside_cell(self) -> tuple[int, int]:
        center_r, center_c = self._center()
        min_r = max(1, self._config.outside_min_radius)
        max_r = max(min_r, self._config.outside_max_radius)

        # Sample radius uniformly in [min_r, max_r]
        radius = self._rng.integers(min_r, max_r + 1)
        # Sample angle uniformly
        theta = self._rng.random() * 2 * math.pi
        dr = int(round(radius * math.sin(theta)))
        dc = int(round(radius * math.cos(theta)))
        return center_r + dr, center_c + dc

    def _place_outside_objects(self, grid: np.ndarray) -> None:
        outside_symbols: list[str] = []
        for name, count in self._config.outside_objects.items():
            outside_symbols.extend([name] * count)

        if not outside_symbols:
            return

        H, W = self._config.height, self._config.width
        max_attempts = 1000
        for sym in outside_symbols:
            placed = False
            for _ in range(max_attempts):
                r, c = self._random_outside_cell()
                if not self._in_bounds(r, c):
                    continue
                if grid[r, c] == "empty":
                    grid[r, c] = sym
                    placed = True
                    break
            if not placed:
                # Fallback: scan for any empty cell at or beyond min radius
                center_r, center_c = self._center()
                min_r = max(1, self._config.outside_min_radius)
                found = False
                for rr in range(self._config.border_width, H - self._config.border_width):
                    for cc in range(self._config.border_width, W - self._config.border_width):
                        if grid[rr, cc] != "empty":
                            continue
                        if abs(rr - center_r) + abs(cc - center_c) >= min_r:
                            grid[rr, cc] = sym
                            found = True
                            break
                    if found:
                        break

    def build(self):
        # Reset RNG to ensure deterministic builds across multiple calls
        if self._config.seed is not None:
            self._rng = np.random.default_rng(self._config.seed)

        grid = np.full((self._config.height, self._config.width), "empty", dtype="<U50")

        # Draw border first if needed
        if self._config.border_width > 0:
            draw_border(grid, self._config.border_width, self._config.border_object)

        # Place the agent at center (single-agent instance)
        if self._config.center_agent:
            r, c = self._center()
            if self._in_bounds(r, c):
                grid[r, c] = "agent.agent"

        # Place hub and outside objects
        self._place_hub_objects(grid)
        self._place_outside_objects(grid)

        return GameMap(grid)
