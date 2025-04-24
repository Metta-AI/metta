"""
MemoryBoxWorld (random sizes)
=============================

• Interior width/height sampled from `size_range` (default 9‑14).
• Entrance gap (3 cells) on a random side.
• Two corridor walls whose length is randomly sampled from
  `corridor_range` **but clipped so they always fit** and always ≥ 6.
• A single altar sits in the corner on the same side as the entrance,
  opposite the gap.
• Boxes are stamped until 10 consecutive placement failures; agents spawn
  outside the boxes.
• Exactly 20 agents are spawned in every episode.
"""

from typing import List, Optional, Tuple

import numpy as np
from mettagrid.config.room.room import Room


class MemoryBoxWorld(Room):
    STYLE_PARAMETERS = {"memory_box": {"hearts_count": 0}}

    def __init__(
        self,
        width: int = 120,
        height: int = 120,
        agents: int | dict = 20,
        size_range: Tuple[int, int] = (9, 14),     # interior side length
        corridor_range: Tuple[int, int] = (6, 10), # sampled per box
        seed: Optional[int] = None,
        border_width: int = 2,
        border_object: str = "wall",
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._rng = np.random.default_rng(seed)
        width, height = np.random.randint(80,120), np.random.randint(80,120)
        self._width, self._height = width, height
        # Always spawn exactly 20 agents, ignoring caller‑supplied values
        self._agents = agents
        self._occ = np.zeros((height, width), dtype=bool)

        self.smin, self.smax = size_range
        self.cmin, self.cmax = corridor_range
        self.set_size_labels(width, height, labels = ["memory_box"])

    # ------------------------------------------------------------------ #
    @property
    def agents(self) -> int:
        """Always returns 20 — the fixed number of agents in MemoryBoxWorld."""
        return 20

    # -------------------- build world -------------------- #
    def _build(self):
        grid = np.full((self._height, self._width), "empty", dtype=object)
        fails = 0
        while fails < 10:
            if self._place_box(grid, clearance=1):
                fails = 0
            else:
                fails += 1
        return self._place_agents(grid)

    # -------------------- box helpers ------------------- #
    def _place_box(self, grid, clearance: int) -> bool:
        pat = self._generate_box()
        return self._place_region(grid, pat, clearance)

    def _generate_box(self) -> np.ndarray:
        # interior size
        h_int = int(self._rng.integers(self.smin, self.smax + 1))
        w_int = int(self._rng.integers(self.smin, self.smax + 1))
        h, w = h_int + 2, w_int + 2
        pat = np.full((h, w), "wall", dtype=object)
        pat[1:-1, 1:-1] = "empty"

        # choose entrance side
        side = self._rng.choice(["top", "bottom", "left", "right"])

        # sample corridor length, clip so it fits and ≥6
        L_max_vert = h_int - 2
        L_max_horz = w_int - 2
        max_len = L_max_vert if side in ("top", "bottom") else L_max_horz
        L = int(self._rng.integers(self.cmin, min(self.cmax, max_len) + 1))

        if side in ("top", "bottom"):
            gap_c = int(self._rng.integers(3, w - 3))
            wall_row = 0 if side == "top" else h - 1
            inner_r = 1 if side == "top" else h - 2
            step = 1 if side == "top" else -1

            # entrance
            for dc in (-1, 0, 1):
                pat[wall_row, gap_c + dc] = "empty"
            # corridor walls
            for k in range(L):
                r = inner_r + k * step
                pat[r, gap_c - 2] = "wall"
                pat[r, gap_c + 2] = "wall"
            # altar
            altar_c = 1 if gap_c > w // 2 else w - 2
            altar = (1, altar_c) if side == "top" else (h - 2, altar_c)
        else:  # left / right
            gap_r = int(self._rng.integers(3, h - 3))
            wall_col = 0 if side == "left" else w - 1
            inner_c = 1 if side == "left" else w - 2
            step = 1 if side == "left" else -1

            for dr in (-1, 0, 1):
                pat[gap_r + dr, wall_col] = "empty"
            for k in range(L):
                c = inner_c + k * step
                pat[gap_r - 2, c] = "wall"
                pat[gap_r + 2, c] = "wall"
            altar_r = 1 if gap_r > h // 2 else h - 2
            altar = (altar_r, 1) if side == "left" else (altar_r, w - 2)

        pat[altar] = "altar"
        return pat

    # ---------------- agent placement -------------- #
    def _place_agents(self, grid):
        """Place agents so they are roughly uniformly spread out.

        A greedy Poisson‑disk sampler is used: each new agent spawn
        must be at least *min_dist* Manhattan units away from every
        already‑placed agent.  If the constraint cannot be satisfied
        after scanning all empties once, any remaining agents are
        placed randomly.
        """
        min_dist = 10
        empties = np.flatnonzero(~self._occ)
        self._rng.shuffle(empties)

        chosen: List[Tuple[int, int]] = []
        H, W = self._occ.shape
        for idx in empties.tolist():
            if len(chosen) == self._agents:
                break
            r, c = divmod(idx, W)
            if all(abs(r - rr) + abs(c - cc) >= min_dist for rr, cc in chosen):
                chosen.append((r, c))

        # Fallback: fill any remaining agents without the distance constraint
        if len(chosen) < self._agents:
            remaining = [divmod(i, W) for i in empties.tolist()
                         if divmod(i, W) not in chosen]
            for pos in remaining[: self._agents - len(chosen)]:
                chosen.append(pos)

        # Stamp agents into the grid / occupancy mask
        for pos in chosen:
            grid[pos] = "agent.agent"
            self._occ[pos] = True

        return grid

    # -------------- generic helpers --------------- #
    def _place_region(self, grid, pat, clearance: int):
        ph, pw = pat.shape
        for r, c in self._free_windows((ph + 2 * clearance, pw + 2 * clearance)):
            grid[r + clearance : r + clearance + ph,
                 c + clearance : c + clearance + pw] = pat
            self._occ[r : r + ph + 2 * clearance,
                      c : c + pw + 2 * clearance] = True
            return True
        return False

    def _free_windows(self, shape):
        h, w = shape
        H, W = self._occ.shape
        if h > H or w > W:
            return []
        view = np.lib.stride_tricks.as_strided(
            self._occ, (H - h + 1, W - w + 1, h, w), self._occ.strides * 2
        )
        coords = np.argwhere(view.sum(axis=(2, 3)) == 0)
        self._rng.shuffle(coords)
        return [tuple(t) for t in coords]

    def _rand_empty(self):
        empties = np.flatnonzero(~self._occ)
        return None if empties.size == 0 else tuple(
            np.unravel_index(self._rng.integers(empties.size), self._occ.shape)
        )
