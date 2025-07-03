"""
SnakeyCylinder multi-agent room.

Generates a network of 1-tile-wide tubes (cylinders) that snake around the map.
• Tubes are carved by a biased random walk, segment length 4-12.
• Random turns create a winding network; a handful of exterior openings are carved as entry points.
• Passing bays: with small probability, a side alcove cell is carved (still sealed externally) so agents can step aside.
• Objects (altar / mine_red / ore_red) are placed inside widened pockets that do not block the main tube –
  a 3×3 chamber is carved around the object, centred on the tube cell.
• Agents are then placed on remaining empty cells; agent count is configurable.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room

# -----------------------------------------------------------------------------


class Manhatten(Room):
    """Grid-like interlocking corridor environment (Manhatten style)."""

    DIRS: list[Tuple[int, int]] = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N,E,S,W

    def __init__(
        self,
        width: int,
        height: int,
        objects: DictConfig,
        agents: int = 16,
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: str = "wall",
        turn_prob: float = 0.15,
        alcove_prob: float = 0.15,
        corridor_spacing: int | None = None,
        heart_prob: float = 0.1,
        team: str = "agent",
    ) -> None:
        super().__init__(border_width=border_width, border_object=border_object, labels=["snakey_cylinder"])
        self.set_size_labels(width, height)
        self.W, self.H = width, height
        self._agents_n = agents
        self._objects_cfg = objects
        self.rng = np.random.default_rng(seed)
        self.turn_prob = turn_prob
        self.alcove_prob = alcove_prob
        self.corridor_spacing = corridor_spacing
        self.heart_prob = heart_prob
        self._team = team

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_int(val) -> int:
        try:
            return int(val)
        except Exception:
            return 0

    def _object_counts(self) -> Dict[str, int]:
        return {k: self._to_int(self._objects_cfg.get(k, 0)) for k in ("altar", "mine_red", "generator_red")}

    # ------------------------------------------------------------------
    # Build main grid (interlocking corridors)
    # ------------------------------------------------------------------

    def _build(self) -> np.ndarray:  # type: ignore[override]
        grid = np.full((self.H, self.W), "wall", dtype=object)

        # Decide corridor spacing (ensure at least 4 tiles apart)
        spacing = self.corridor_spacing if self.corridor_spacing is not None else max(6, min(self.W, self.H) // 6)
        off_r = int(self.rng.integers(2, spacing))
        off_c = int(self.rng.integers(2, spacing))

        horiz_rows = list(range(off_r, self.H - 1, spacing))
        vert_cols = list(range(off_c, self.W - 1, spacing))

        # carve horizontal corridors
        for r in horiz_rows:
            grid[r, 1 : self.W - 1] = "empty"

        # carve vertical corridors
        for c in vert_cols:
            grid[1 : self.H - 1, c] = "empty"

        # Collect walkway cells
        walkway_cells = [(r, c) for r in horiz_rows for c in range(1, self.W - 1)]
        walkway_cells += [(r, c) for c in vert_cols for r in range(1, self.H - 1)]

        # Passing alcoves inside walls
        alcove_cells: list[Tuple[int, int]] = []
        for r, c in walkway_cells:
            if self.rng.random() < self.alcove_prob:
                dirs = self.DIRS.copy()
                self.rng.shuffle(dirs)
                for dr, dc in dirs:
                    rr, cc = r + dr, c + dc
                    if 0 < rr < self.H - 1 and 0 < cc < self.W - 1 and grid[rr, cc] == "wall":
                        grid[rr, cc] = "empty"
                        alcove_cells.append((rr, cc))
                        break

        # Entry points: open four random border cells adjacent to corridors
        self.rng.shuffle(walkway_cells)
        entries = 0
        for r, c in walkway_cells:
            if entries >= 4:
                break
            if r == horiz_rows[0]:
                grid[0, c] = "empty"
                entries += 1
            elif r == horiz_rows[-1]:
                grid[self.H - 1, c] = "empty"
                entries += 1
            elif c == vert_cols[0]:
                grid[r, 0] = "empty"
                entries += 1
            elif c == vert_cols[-1]:
                grid[r, self.W - 1] = "empty"
                entries += 1

        # Place objects at crossroads (intersections of corridors)
        crossroads = [(r, c) for r in horiz_rows for c in vert_cols]
        self.rng.shuffle(crossroads)
        placement_cells = crossroads + alcove_cells
        self.rng.shuffle(placement_cells)

        counts = self._object_counts()
        obj_list: list[str] = []
        for name, cnt in counts.items():
            obj_list.extend([name] * cnt)
        self.rng.shuffle(obj_list)

        for obj, (r, c) in zip(obj_list, placement_cells, strict=False):
            # widen crossroads by carving immediate neighbours (keep 3x3 open)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.H and 0 <= cc < self.W:
                        grid[rr, cc] = "empty"
            grid[r, c] = obj

        # Place agents on remaining empty cells
        empties = list(zip(*np.where(grid == "empty"), strict=False))
        self.rng.shuffle(empties)
        if len(empties) < self._agents_n:
            raise ValueError("Not enough empty tiles for agents; reduce number or enlarge map.")
        for ar, ac in empties[: self._agents_n]:
            grid[ar, ac] = f"agent.{self._team}"

        # Optionally place hearts in remaining unused alcoves
        remaining_alcoves = [cell for cell in alcove_cells if grid[cell[0], cell[1]] == "empty"]
        for r, c in remaining_alcoves:
            if self.rng.random() < self.heart_prob:
                grid[r, c] = "heart"

        return grid


# Alias for backward compatibility
SnakeyCylinder = Manhatten
SnakeyCylinderRoom = Manhatten
