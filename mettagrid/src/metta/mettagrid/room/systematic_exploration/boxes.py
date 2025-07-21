"""
Boxes Terrain
=============
• Hollow rectangles (3–12 × 2–12) with crenellated walls + one doorway
• 1–3 altars inside each box, each surrounded by walls
• Always spawns exactly `agents` agents (fallback to any empty tile)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class BoxesTerrain(Room):
    def __init__(
        self,
        width: int = 160,
        height: int = 160,
        agents: int = 4,
        objects: DictConfig | dict | None = None,
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: str = "wall",
        #
        num_boxes: int = 20,
        box_width: int = 7,
        box_height: int = 7,
        altars_per_box: int = 2,
        gap: int = 1,
    ) -> None:
        super().__init__(border_width=border_width, border_object=border_object, labels=["boxes"])
        self.set_size_labels(width, height)

        self.H, self.W = height, width
        self.rng = np.random.default_rng(seed)
        self.n_agents = int(agents)
        self.user_objects = objects if isinstance(objects, dict) else {}
        self.occ = np.zeros((height, width), dtype=bool)

        self.n_boxes = int(num_boxes)
        self.box_width = int(box_width)
        self.box_height = int(box_height)
        self.altars_per_box = int(altars_per_box)
        self.gap = max(0, int(gap))

    # ------------------------------------------------------------------ #
    # Build                                                               #
    # ------------------------------------------------------------------ #
    def _build(self) -> np.ndarray:
        grid = np.full((self.H, self.W), "empty", dtype=object)

        # border
        grid[0, :], grid[-1, :] = "wall", "wall"
        grid[:, 0], grid[:, -1] = "wall", "wall"
        self.occ[0, :], self.occ[-1, :] = True, True
        self.occ[:, 0], self.occ[:, -1] = True, True

        boxes: List[Tuple[int, int, int, int]] = []

        # 1 ─ place boxes
        tries = 0
        while len(boxes) < self.n_boxes and tries < 4000:
            tries += 1
            bw = self.box_width
            bh = self.box_height
            r0 = int(self.rng.integers(1 + self.gap, self.H - bh - 1 - self.gap))
            c0 = int(self.rng.integers(1 + self.gap, self.W - bw - 1 - self.gap))
            if self.occ[r0 - self.gap : r0 + bh + self.gap, c0 - self.gap : c0 + bw + self.gap].any():
                continue

            # outer walls
            grid[r0, c0 : c0 + bw] = "wall"
            grid[r0 + bh - 1, c0 : c0 + bw] = "wall"
            grid[r0 : r0 + bh, c0] = "wall"
            grid[r0 : r0 + bh, c0 + bw - 1] = "wall"

            # crenellate
            rr, cc = np.indices((bh, bw))
            mask = (rr + cc) % 2 == 1
            grid[r0, c0 : c0 + bw][mask[0]] = "empty"
            grid[r0 + bh - 1, c0 : c0 + bw][mask[-1]] = "empty"
            grid[r0 : r0 + bh, c0][mask[:, 0]] = "empty"
            grid[r0 : r0 + bh, c0 + bw - 1][mask[:, -1]] = "empty"

            # doorway
            if bw > 2 and bh > 2:
                side = self.rng.choice(["top", "bottom", "left", "right"])
                if side in ("top", "bottom"):
                    cd = int(self.rng.integers(c0 + 1, c0 + bw - 1))
                    rd = r0 if side == "top" else r0 + bh - 1
                    grid[rd, cd] = "empty"
                else:
                    rd = int(self.rng.integers(r0 + 1, r0 + bh - 1))
                    cd = c0 if side == "left" else c0 + bw - 1
                    grid[rd, cd] = "empty"

            self.occ[grid == "wall"] = True
            boxes.append((r0, c0, bh, bw))

        # 2 ─ altars inside
        for r0, c0, bh, bw in boxes:
            viable = [
                (r, c)
                for r in range(r0 + 2, r0 + bh - 2)
                for c in range(c0 + 2, c0 + bw - 2)
                if self._can_place_altar_with_walls(r, c, grid)
            ]
            self.rng.shuffle(viable)
            for r, c in viable[: self.altars_per_box]:
                grid[r, c] = "altar"
                self.occ[r, c] = True
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if grid[nr, nc] == "empty":
                            grid[nr, nc] = "wall"
                            self.occ[nr, nc] = True

        # 3 ─ place agents (safe then fallback)
        remaining = self.n_agents
        while remaining:
            pos = self._pick_safe_empty()
            if pos is None:
                break
            grid[pos] = "agent.agent"
            self.occ[pos] = True
            remaining -= 1
        while remaining:
            flat = np.flatnonzero(~self.occ)
            if flat.size == 0:
                break
            pos = np.unravel_index(self.rng.integers(flat.size), self.occ.shape)
            grid[pos] = "agent.agent"
            self.occ[pos] = True
            remaining -= 1
        if remaining:
            raise RuntimeError("Unable to place all agents (map too cluttered).")

        # 4 ─ extras
        for name, cnt in self.user_objects.items():
            for _ in range(int(cnt)):
                pos = self._pick_safe_empty()
                if pos is None:
                    break
                grid[pos] = name
                self.occ[pos] = True

        return grid

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _can_place_altar_with_walls(self, r: int, c: int, grid: np.ndarray) -> bool:
        if grid[r, c] != "empty":
            return False
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if grid[nr, nc] not in ("empty", "wall"):
                    return False
        return True

    def _pick_safe_empty(self) -> Optional[Tuple[int, int]]:
        empty = ~self.occ
        if not empty.any():
            return None
        safe = empty.copy()
        for dr, dc in DIRS:
            safe &= ~np.roll(self.occ, shift=(dr, dc), axis=(0, 1))
        safe[0, :] = safe[-1, :] = False
        safe[:, 0] = safe[:, -1] = False
        flat = np.flatnonzero(safe)
        if flat.size == 0:
            return None
        pos = np.unravel_index(self.rng.integers(flat.size), safe.shape)
        return pos
