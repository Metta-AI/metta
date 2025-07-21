"""
Spindly-Spider Terrain
======================
A variant of SpiderTerrain with very long, randomly turning / branching
legs.

Key differences:
  • Legs draw a self-avoiding walk with occasional 90° turns.
  • At random points they *branch*, spawning shorter sub-legs.
  • Default legs are much longer (see YAML).

Everything else—body placement, gap clearance, altar placement—matches the
standard spiders environment, so it plugs straight into MettaGrid.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room

# ────────────────────────────────────────────────────────────────────────── #
# Helper utilities                                                          #
# ────────────────────────────────────────────────────────────────────────── #
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E


def _perpendicular(dr: int, dc: int) -> List[Tuple[int, int]]:
    """Return the two directions 90° to (dr, dc)."""
    if dr != 0:  # vertical → left/right
        return [(0, -1), (0, 1)]
    else:  # horizontal → up/down
        return [(-1, 0), (1, 0)]


# ────────────────────────────────────────────────────────────────────────── #
# Terrain class                                                             #
# ────────────────────────────────────────────────────────────────────────── #
class SpindlySpiderTerrain(Room):
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
        num_spiders: int | Tuple[int, int] = 6,
        body_lo: int = 4,
        body_hi: int = 10,
        legs_lo: int = 6,
        legs_hi: int = 12,
        leg_length: int = 15,
        branch_prob: float = 0.15,  # probability to branch each step
        turn_prob: float = 0.25,  # probability to turn 90°
        gap: int = 1,
        hearts_per_spider: Tuple[int, int] = (1, 2),  # altars per spider
    ) -> None:
        super().__init__(border_width=border_width, border_object=border_object, labels=["spindly_spiders"])
        self.set_size_labels(width, height)

        # Core state
        self.H, self.W = height, width
        self.rng = np.random.default_rng(seed)
        self.agents = int(agents)
        self.extra_objects = objects or {}
        self.occ = np.zeros((height, width), dtype=bool)

        # Parameters
        self.num_spiders = num_spiders
        self.body_lo, self.body_hi = body_lo, body_hi
        self.legs_lo, self.legs_hi = legs_lo, legs_hi
        self.len_lo, self.len_hi = leg_length, leg_length
        self.branch_prob = float(branch_prob)
        self.turn_prob = float(turn_prob)
        self.gap = max(0, int(gap))
        self.alt_lo, self.alt_hi = int(hearts_per_spider[0]), int(hearts_per_spider[1])

    # ────────────────────────────────────────────────────────────────── #
    # Build                                                               #
    # ────────────────────────────────────────────────────────────────── #
    def _build(self) -> np.ndarray:
        grid = np.full((self.H, self.W), "empty", dtype=object)

        # Border walls
        grid[0, :], grid[-1, :] = "wall", "wall"
        grid[:, 0], grid[:, -1] = "wall", "wall"
        self.occ[0, :], self.occ[-1, :] = True, True
        self.occ[:, 0], self.occ[:, -1] = True, True

        bodies: List[Tuple[int, int, int, int]] = []

        # 1 ─ place bodies with gap
        tries = 0
        while len(bodies) < self.num_spiders and tries < 5000:
            tries += 1
            h = int(self.rng.integers(self.body_lo, self.body_hi + 1))
            w = int(self.rng.integers(self.body_lo, self.body_hi + 1))
            r0 = int(self.rng.integers(1 + self.gap, self.H - h - 1 - self.gap))
            c0 = int(self.rng.integers(1 + self.gap, self.W - w - 1 - self.gap))
            if self.occ[r0 - self.gap : r0 + h + self.gap, c0 - self.gap : c0 + w + self.gap].any():
                continue
            grid[r0 : r0 + h, c0 : c0 + w] = "wall"
            self.occ[r0 : r0 + h, c0 : c0 + w] = True
            bodies.append((r0, c0, h, w))

        # 2 ─ grow spindly legs
        for r0, c0, h, w in bodies:
            n_legs = int(self.rng.integers(self.legs_lo, self.legs_hi + 1))
            for _ in range(n_legs):
                # choose starting point and initial direction
                if self.rng.random() < 0.5:  # top/bottom
                    c = int(self.rng.integers(c0, c0 + w))
                    r = r0 if self.rng.random() < 0.5 else r0 + h - 1
                    dr, dc = (-1, 0) if r == r0 else (1, 0)
                else:  # left/right
                    r = int(self.rng.integers(r0, r0 + h))
                    c = c0 if self.rng.random() < 0.5 else c0 + w - 1
                    dr, dc = (0, -1) if c == c0 else (0, 1)

                goal_len = int(self.rng.integers(self.len_lo, self.len_hi + 1))
                self._grow_leg(grid, r, c, dr, dc, goal_len, depth=0)

        # 3 ─ place altars hugging bodies
        candidates: List[Tuple[int, int]] = []
        for r0, c0, h, w in bodies:
            for rr in range(r0 - 1, r0 + h + 1):
                for cc in range(c0 - 1, c0 + w + 1):
                    if not (1 <= rr < self.H - 1 and 1 <= cc < self.W - 1):
                        continue
                    if abs(rr - r0) in {0, h - 1} or abs(cc - c0) in {0, w - 1}:
                        if not self.occ[rr, cc]:
                            candidates.append((rr, cc))
        self.rng.shuffle(candidates)
        total_altars = sum(int(self.rng.integers(self.alt_lo, self.alt_hi + 1)) for _ in bodies)
        for rr, cc in candidates[:total_altars]:
            grid[rr, cc] = "altar"
            self.occ[rr, cc] = True

        # 4 ─ agents
        for _ in range(self.agents):
            pos = self._pick_empty()
            if pos is None:
                break
            rr, cc = pos
            grid[rr, cc] = "agent.agent"
            self.occ[rr, cc] = True

        # 5 ─ user objects
        for name, cnt in self.extra_objects.items():
            for _ in range(int(cnt)):
                pos = self._pick_empty()
                if pos is None:
                    break
                rr, cc = pos
                grid[rr, cc] = name
                self.occ[rr, cc] = True

        return grid

    # ────────────────────────────────────────────────────────────────── #
    # Leg growth                                                         #
    # ────────────────────────────────────────────────────────────────── #
    def _grow_leg(
        self,
        grid: np.ndarray,
        r: int,
        c: int,
        dr: int,
        dc: int,
        remaining: int,
        depth: int,
    ) -> None:
        """Recursive self-avoiding leg with branching."""
        if remaining <= 0 or depth > 5:
            return
        rr, cc = r + dr, c + dc
        if not (1 <= rr < self.H - 1 and 1 <= cc < self.W - 1):
            return
        if grid[rr, cc] == "wall":  # hit another wall
            return

        # mark this tile
        grid[rr, cc] = "wall"
        self.occ[rr, cc] = True

        # maybe branch
        if self.rng.random() < self.branch_prob:
            br_dr, br_dc = self.rng.choice(_perpendicular(dr, dc))
            br_len = max(1, remaining // 2)
            self._grow_leg(grid, rr, cc, br_dr, br_dc, br_len, depth + 1)

        # maybe turn
        if self.rng.random() < self.turn_prob:
            dr, dc = self.rng.choice(_perpendicular(dr, dc))

        # continue straight
        self._grow_leg(grid, rr, cc, dr, dc, remaining - 1, depth)

    # ────────────────────────────────────────────────────────────────── #
    # Helper                                                             #
    # ────────────────────────────────────────────────────────────────── #
    def _pick_empty(self) -> Optional[Tuple[int, int]]:
        empty = np.flatnonzero(~self.occ)
        if empty.size == 0:
            return None
        idx = self.rng.integers(empty.size)
        return np.unravel_index(idx, self.occ.shape)
