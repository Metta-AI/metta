"""
Spider-Terrain
==============
Generates a maze of “spiders”:

• **Body** – solid wall rectangle (size sampled per spider)
• **Legs** – straight wall lines growing from random body edges
• **Altars** – placed right against each body’s perimeter (never adrift);
              each altar starts with 1 heart and a long cooldown

Sweep-able parameters (see YAML):
    num_spiders        – total spiders
    body_size          – [min,max] body side length
    legs_per_spider    – [min,max] legs per spider
    leg_length         – [min,max] length of a leg
    gap                – minimum clearance between bodies
    hearts_per_spider  – [min,max] **altars** to place per spider
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
from omegaconf import DictConfig, ListConfig
from metta.mettagrid.room.room import Room


class SpiderTerrain(Room):
    # ------------------------------------------------------------------ #
    # Init                                                                #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        width: int = 160,
        height: int = 160,
        agents: int = 1,
        objects: DictConfig | dict | None = None,
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: str = "wall",
        #
        num_spiders: int = 8,
        body_size: int = 6,
        legs_per_spider: int = 6,
        leg_length: int = 12,
        gap: int = 1,
        hearts_per_spider: int = 2,   # ALTARS per spider
    ) -> None:
        super().__init__(
            border_width=border_width,
            border_object=border_object,
            labels=["spiders"],
        )
        self.set_size_labels(width, height)

        # RNG & arena bookkeeping
        self._H, self._W = height, width
        self._rng = np.random.default_rng(seed)
        self._agents = int(agents)
        self._extra_objects = objects or {}
        self._occ = np.zeros((height, width), dtype=bool)

        # --- resolve parameters -----------------------------------------
        self._num_spiders = int(num_spiders)
        self._body_size = int(body_size)
        self._legs_per_spider = int(legs_per_spider)
        self._leg_length = int(leg_length)
        self._gap = max(0, int(gap))
        self._hearts_per_spider = int(hearts_per_spider)

    # ------------------------------------------------------------------ #
    # Build                                                               #
    # ------------------------------------------------------------------ #
    def _build(self) -> np.ndarray:
        grid = np.full((self._H, self._W), "empty", dtype=object)

        # 0 ─ border walls
        grid[0, :], grid[-1, :] = "wall", "wall"
        grid[:, 0], grid[:, -1] = "wall", "wall"
        self._occ[0, :], self._occ[-1, :] = True, True
        self._occ[:, 0], self._occ[:, -1] = True, True

        # 1 ─ spider bodies
        bodies: list[Tuple[int, int, int, int]] = []
        attempts = 0
        while len(bodies) < self._num_spiders and attempts < 5000:
            attempts += 1
            h = self._body_size
            w = self._body_size
            r0 = int(self._rng.integers(1 + self._gap, self._H - h - 1 - self._gap))
            c0 = int(self._rng.integers(1 + self._gap, self._W - w - 1 - self._gap))

            # ensure clearance
            if self._occ[r0 - self._gap : r0 + h + self._gap,
                         c0 - self._gap : c0 + w + self._gap].any():
                continue

            grid[r0 : r0 + h, c0 : c0 + w] = "wall"
            self._occ[r0 : r0 + h, c0 : c0 + w] = True
            bodies.append((r0, c0, h, w))

        # 2 ─ legs
        for r0, c0, h, w in bodies:
            n_legs = self._legs_per_spider
            for _ in range(n_legs):
                if self._rng.random() < 0.5:           # pick top or bottom edge
                    c = int(self._rng.integers(c0, c0 + w))
                    r = r0 if self._rng.random() < 0.5 else r0 + h - 1
                    dr, dc = (-1, 0) if r == r0 else (1, 0)
                else:                                  # left or right edge
                    r = int(self._rng.integers(r0, r0 + h))
                    c = c0 if self._rng.random() < 0.5 else c0 + w - 1
                    dr, dc = (0, -1) if c == c0 else (0, 1)

                length = self._leg_length
                for step in range(1, length + 1):
                    rr, cc = r + dr * step, c + dc * step
                    if not (1 <= rr < self._H - 1 and 1 <= cc < self._W - 1):
                        break
                    if grid[rr, cc] == "wall":
                        break
                    grid[rr, cc] = "wall"
                    self._occ[rr, cc] = True

        # 3 ─ altars hugging body perimeter
        candidate_cells: List[Tuple[int, int]] = []
        for r0, c0, h, w in bodies:
            for rr in range(r0 - 1, r0 + h + 1):
                for cc in range(c0 - 1, c0 + w + 1):
                    if not (1 <= rr < self._H - 1 and 1 <= cc < self._W - 1):
                        continue
                    if abs(rr - r0) in {0, h - 1} or abs(cc - c0) in {0, w - 1}:
                        if not self._occ[rr, cc]:
                            candidate_cells.append((rr, cc))

        self._rng.shuffle(candidate_cells)
        total_altars = self._hearts_per_spider * len(bodies)
        for rr, cc in candidate_cells[:total_altars]:
            grid[rr, cc] = "altar"
            self._occ[rr, cc] = True

        # 4 ─ agents
        for _ in range(self._agents):
            pos = self._pick_empty()
            if pos is None:
                break
            rr, cc = pos
            grid[rr, cc] = "agent.agent"
            self._occ[rr, cc] = True

        # 5 ─ extra user objects
        for name, cnt in self._extra_objects.items():
            for _ in range(int(cnt)):
                pos = self._pick_empty()
                if pos is None:
                    break
                rr, cc = pos
                grid[rr, cc] = name
                self._occ[rr, cc] = True

        return grid

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _pick_empty(self) -> Optional[Tuple[int, int]]:
        empty_flat = np.flatnonzero(~self._occ)
        if empty_flat.size == 0:
            return None
        idx = self._rng.integers(empty_flat.size)
        return np.unravel_index(idx, self._occ.shape)
