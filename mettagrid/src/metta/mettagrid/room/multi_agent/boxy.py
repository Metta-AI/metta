"""Boxy multi-agent environment.
Large arena with multiple square wall boxes each containing altars.
Entrance to each box is a 1-tile wide corridor flanked by parallel walls that
extends 3-6 tiles into the outer area. Mines and generators are scattered
around the outer area. All mines start with one ore. Altars and generators
start empty.  Agents spawn in empty tiles across the whole map.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room


class Boxy(Room):
    """Multi-agent environment with altar boxes and outer resources."""

    def __init__(
        self,
        width: int,
        height: int,
        objects: DictConfig | None = None,
        agents: int = 20,
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: str = "wall",
        team: str = "agent",
    ) -> None:
        super().__init__(border_width=border_width, border_object=border_object, labels=["boxy"])
        self.set_size_labels(width, height)
        self._w = width
        self._h = height
        self._agents_n = agents
        self._objects_cfg = objects or DictConfig({})
        self._rng = np.random.default_rng(seed)
        self._team = team

    # ------------------------------------------------------------------
    def _to_int(self, val, default: int) -> int:
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    def _build(self) -> np.ndarray:  # type: ignore[override]
        grid = np.full((self._h, self._w), "empty", dtype=object)
        self._occ = np.zeros((self._h, self._w), dtype=bool)

        # Parameters from YAML (with sensible fallbacks)
        mine_total = self._to_int(self._objects_cfg.get("mine_red", 30), 30)
        generator_total = self._to_int(self._objects_cfg.get("generator_red", 30), 30)

        # Determine number of boxes (and therefore altars)
        altar_cfg = self._objects_cfg.get("altar", None)
        if altar_cfg is not None:
            num_boxes = self._to_int(altar_cfg, 12)
        else:
            num_boxes = int(self._rng.integers(10, 15))  # default 10-14 boxes

        interior_coords: List[Tuple[int, int]] = []  # all interior cells (excluding walls)

        box_attempts = 0
        placed_boxes = 0
        while placed_boxes < num_boxes and box_attempts < 500:
            box_attempts += 1
            size = int(self._rng.integers(4, 10))  # 4 ‑ 9 inclusive
            top = int(self._rng.integers(1, self._h - size - 1))
            left = int(self._rng.integers(1, self._w - size - 1))

            # Check overlap with existing structures, leaving 1-tile buffer
            overlap = False
            for r in range(top - 1, top + size + 1):
                for c in range(left - 1, left + size + 1):
                    if r < 0 or c < 0 or r >= self._h or c >= self._w:
                        overlap = True
                        break
                    if self._occ[r, c]:
                        overlap = True
                        break
                if overlap:
                    break
            if overlap:
                continue

            # Place perimeter walls
            for r in range(top, top + size):
                for c in range(left, left + size):
                    if r in (top, top + size - 1) or c in (left, left + size - 1):
                        grid[r, c] = "wall"
                        self._occ[r, c] = True
            # Store interior coords and select a random cell for an altar
            box_interior: List[Tuple[int, int]] = []
            for r in range(top + 1, top + size - 1):
                for c in range(left + 1, left + size - 1):
                    box_interior.append((r, c))
                    interior_coords.append((r, c))

            # Place exactly one altar inside this box
            self._rng.shuffle(box_interior)
            if box_interior:
                ar, ac = box_interior[0]
                grid[ar, ac] = "altar"
                self._occ[ar, ac] = True

            # Create an entrance
            self._create_entrance(grid, top, left, size)

            placed_boxes += 1

        # ---- scatter mines and generator_reds outside boxes ----
        interior_set = set(interior_coords)
        outer_coords = [
            (r, c)
            for r in range(self._h)
            for c in range(self._w)
            if grid[r, c] == "empty" and (r, c) not in interior_set
        ]
        self._rng.shuffle(outer_coords)

        for r, c in outer_coords[:mine_total]:
            grid[r, c] = "mine_red"
            self._occ[r, c] = True
        for r, c in outer_coords[mine_total : mine_total + generator_total]:
            grid[r, c] = "generator_red"
            self._occ[r, c] = True

        # ---- place agents ----
        empty_cells = [(r, c) for r in range(self._h) for c in range(self._w) if grid[r, c] == "empty"]
        self._rng.shuffle(empty_cells)
        for r, c in empty_cells[: self._agents_n]:
            grid[r, c] = f"agent.{self._team}"
            self._occ[r, c] = True

        return grid

    # ------------------------------------------------------------------
    def _create_entrance(self, grid: np.ndarray, top: int, left: int, size: int) -> None:
        """Carve an entrance corridor to the given box."""
        side = int(self._rng.integers(0, 4))  # 0=top,1=bottom,2=left,3=right
        corridor_len = int(self._rng.integers(3, 7))  # 3-6 inclusive

        if side == 0:  # top side
            row = top
            col = int(self._rng.integers(left + 1, left + size - 1))
            self._carve_corridor(grid, row, col, dr=-1, dc=0, length=corridor_len)
        elif side == 1:  # bottom
            row = top + size - 1
            col = int(self._rng.integers(left + 1, left + size - 1))
            self._carve_corridor(grid, row, col, dr=1, dc=0, length=corridor_len)
        elif side == 2:  # left
            row = int(self._rng.integers(top + 1, top + size - 1))
            col = left
            self._carve_corridor(grid, row, col, dr=0, dc=-1, length=corridor_len)
        else:  # right
            row = int(self._rng.integers(top + 1, top + size - 1))
            col = left + size - 1
            self._carve_corridor(grid, row, col, dr=0, dc=1, length=corridor_len)

    def _carve_corridor(self, grid: np.ndarray, start_r: int, start_c: int, dr: int, dc: int, length: int) -> None:
        """Carve a 1-tile wide walkway flanked by walls along given direction."""
        # Remove the wall at the box perimeter
        grid[start_r, start_c] = "empty"
        self._occ[start_r, start_c] = False

        r, c = start_r, start_c
        for _ in range(length):
            r += dr
            c += dc
            if not (0 <= r < self._h and 0 <= c < self._w):
                break
            # walkway cell
            grid[r, c] = "empty"
            self._occ[r, c] = False
            # side walls
            perp = (-dc, -dr)  # perpendicular direction vector (left side)
            for sign in (-1, 1):
                sr = r + perp[0] * sign
                sc = c + perp[1] * sign
                if 0 <= sr < self._h and 0 <= sc < self._w and grid[sr, sc] == "empty":
                    grid[sr, sc] = "wall"
                    self._occ[sr, sc] = True
