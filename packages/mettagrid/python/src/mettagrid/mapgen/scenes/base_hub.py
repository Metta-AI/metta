from typing import Literal

import numpy as np

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene


class BaseHubParams(Config):
    altar_object: str = "altar"
    corner_generator: str = "generator_red"
    spawn_symbol: str = "agent.agent"
    include_inner_wall: bool = True


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

        # Corner generators: place near each corner with one-cell clearance
        # Place generators in corners with 1-cell clearance from borders
        gens = [
            (2, 2),  # top-left
            (w - 3, 2),  # top-right
            (2, h - 3),  # bottom-left
            (w - 3, h - 3),  # bottom-right
        ]
        placed = 0
        for x, y in gens:
            if 1 <= x < w - 1 and 1 <= y < h - 1:
                # Ensure 3x3 around is empty
                x0, x1 = max(0, x - 1), min(w, x + 2)
                y0, y1 = max(0, y - 1), min(h, y + 2)
                if np.all(grid[y0:y1, x0:x1] == "empty"):
                    grid[y, x] = self.params.corner_generator
                    placed += 1
        # If not all 4 placed (e.g., too small), try fallback inward spots
        if placed < 4:
            alt = [
                (3, 3),
                (w - 4, 3),
                (3, h - 4),
                (w - 4, h - 4),
            ]
            for x, y in alt:
                if placed >= 4:
                    break
                if 1 <= x < w - 1 and 1 <= y < h - 1 and grid[y, x] == "empty":
                    grid[y, x] = self.params.corner_generator
                    placed += 1

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
