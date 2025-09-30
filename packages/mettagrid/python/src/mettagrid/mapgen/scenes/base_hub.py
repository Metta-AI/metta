from typing import Literal

import numpy as np

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene


class BaseHubParams(Config):
    altar_object: str = "altar"
    corner_generator: str = "generator_red"
    spawn_symbol: str = "agent.agent"
    include_inner_wall: bool = True
    # Order: top-left, top-right, bottom-left, bottom-right.
    corner_objects: list[str] | None = None


class BaseHub(Scene[BaseHubParams]):
    """
    Build a symmetric 11x11 base:
    - Center cell: altar (assembler)
    - Four corner generators with one empty cell of clearance on all sides
    - Symmetric L-shaped empty corridors at each corner to form 4 exits
    - Spawn pads around center with empty clearance
    """

    def render(self):
        grid = self.grid
        h, w = self.height, self.width

        # Fill with empty to start
        grid[:] = "empty"

        cx, cy = w // 2, h // 2

        # Optional inner wall ring around the border of the base area
        if self.params.include_inner_wall and h >= 3 and w >= 3:
            grid[0, :] = "wall"
            grid[-1, :] = "wall"
            grid[:, 0] = "wall"
            grid[:, -1] = "wall"

            # Deterministic 3-wide gates at midpoints of each side
            gate_half = 1
            # top gate centered at cx
            grid[0, cx - gate_half : cx + gate_half + 1] = "empty"
            grid[1, cx - gate_half : cx + gate_half + 1] = "empty"
            # bottom gate
            grid[h - 1, cx - gate_half : cx + gate_half + 1] = "empty"
            grid[h - 2, cx - gate_half : cx + gate_half + 1] = "empty"
            # left gate
            grid[cy - gate_half : cy + gate_half + 1, 0] = "empty"
            grid[cy - gate_half : cy + gate_half + 1, 1] = "empty"
            # right gate
            grid[cy - gate_half : cy + gate_half + 1, w - 1] = "empty"
            grid[cy - gate_half : cy + gate_half + 1, w - 2] = "empty"

        # Place central altar with one-cell clearance.
        # The area is already empty; we keep neighbors empty for access.
        if 1 <= cx < w - 1 and 1 <= cy < h - 1:
            grid[cy, cx] = self.params.altar_object

        # Spawn pads in plus-shape around center with clearance
        spawn_positions = [
            (cx, cy - 2),
            (cx + 2, cy),
            (cx, cy + 2),
            (cx - 2, cy),
        ]
        for x, y in spawn_positions:
            if 1 <= x < w - 1 and 1 <= y < h - 1 and grid[y, x] == "empty":
                grid[y, x] = self.params.spawn_symbol

        # Carve symmetric L-shaped exits in four corners: two-leg corridors 3-wide
        # Ensure each L opens a gap in the inner wall ring to the outside.
        # Top-left L
        self._carve_L(1, 1, orientation="right-down")
        # Top-right L
        self._carve_L(w - 4, 1, orientation="left-down")
        # Bottom-left L
        self._carve_L(1, h - 4, orientation="right-up")
        # Bottom-right L
        self._carve_L(w - 4, h - 4, orientation="left-up")

        # Place corner objects after carving corridors so they are not overwritten
        # Corner order: TL, TR, BL, BR
        primary_positions = [
            (2, 2),
            (w - 3, 2),
            (2, h - 3),
            (w - 3, h - 3),
        ]
        fallback_positions = [
            (3, 3),
            (w - 4, 3),
            (3, h - 4),
            (w - 4, h - 4),
        ]

        names: list[str]
        if self.params.corner_objects and len(self.params.corner_objects) == 4:
            names = list(self.params.corner_objects)
        else:
            names = [self.params.corner_generator] * 4

        remaining: list[tuple[int, int, str]] = []
        # First pass: try primary positions
        for (x, y), name in zip(primary_positions, names, strict=False):
            if 1 <= x < w - 1 and 1 <= y < h - 1:
                x0, x1 = max(0, x - 1), min(w, x + 2)
                y0, y1 = max(0, y - 1), min(h, y + 2)
                if np.all(grid[y0:y1, x0:x1] == "empty"):
                    grid[y, x] = name
                else:
                    remaining.append((x, y, name))
            else:
                remaining.append((x, y, name))

        # Second pass: fallback inward spots for any that couldn't be placed
        for (fx, fy), (_, _, name) in zip(fallback_positions, remaining, strict=False):
            if 1 <= fx < w - 1 and 1 <= fy < h - 1 and grid[fy, fx] == "empty":
                grid[fy, fx] = name

        # Place chest directly below altar after corridors are carved
        if 1 <= cx < w - 1 and 1 <= cy + 1 < h - 1:
            grid[cy + 3, cx] = "chest"

    def _carve_L(self, x: int, y: int, orientation: Literal["right-down", "left-down", "right-up", "left-up"]):
        grid = self.grid
        h, w = self.height, self.width

        width = 3  # corridor thickness
        leg = max(3, min(h, w) // 3)  # leg length based on base size

        def carve_rect(x0: int, y0: int, cw: int, ch: int):
            x1 = max(0, x0)
            y1 = max(0, y0)
            x2 = min(w, x0 + cw)
            y2 = min(h, y0 + ch)
            if x2 > x1 and y2 > y1:
                grid[y1:y2, x1:x2] = "empty"

        if orientation == "right-down":
            # horizontal then vertical
            carve_rect(x, y, leg, width)
            carve_rect(x + leg - width, y, width, leg)
            # open top border
            carve_rect(x, 0, width, 1)
        elif orientation == "left-down":
            carve_rect(x - leg + width, y, leg, width)
            carve_rect(x - leg + width, y, width, leg)
            # open top border
            carve_rect(x - width + 1, 0, width, 1)
        elif orientation == "right-up":
            carve_rect(x, y, leg, width)
            carve_rect(x + leg - width, y - leg + width, width, leg)
            # open left border
            carve_rect(0, y - width + 1, 1, width)
        elif orientation == "left-up":
            carve_rect(x - leg + width, y, leg, width)
            carve_rect(x - leg + width, y - leg + width, width, leg)
            # open bottom border
            carve_rect(x - width + 1, h - 1, width, 1)
