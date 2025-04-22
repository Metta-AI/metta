"""
LabyrinthWorld
==============

A map densely tiled with elongated mini‑mazes.  Each maze …

* is rectangular and elongated (long side = short side + 3–7 cells).
* has one entrance on a short edge.
* hides a single heart **altar** in the diagonally opposite corner.
* is marked fully “occupied”, so agents are spawned only in the open
  corridors **between** labyrinths.

The builder keeps dropping labyrinths until it records 10 consecutive
placement failures, which means the map is essentially full.
"""

from typing import List, Optional, Tuple

import numpy as np

from mettagrid.config.room.room import Room


class LabyrinthWorld(Room):
    STYLE_PARAMETERS = {
        "labyrinth_world": {
            "hearts_count": 0,  # altars live inside labyrinths
            "labyrinths": {"count": 999},
        },
    }

    # ------------------------------------------------------------------ #
    # Initialise
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        width: int,
        height: int,
        agents: int | dict = 0,
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: str = "wall",
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._rng = np.random.default_rng(seed)
        # overall map size chosen randomly if not overridden by YAML
        self._width = width
        self._height = height
        self._agents = agents
        self._occ = np.zeros((height, width), dtype=bool)

    # ------------------------------------------------------------------ #
    # Public build entry
    # ------------------------------------------------------------------ #
    def _build(self) -> np.ndarray:
        return self._build_world()

    # ------------------------------------------------------------------ #
    # World construction
    # ------------------------------------------------------------------ #
    def _build_world(self) -> np.ndarray:
        grid = np.full((self._height, self._width), "empty", dtype=object)

        consecutive_failures = 0
        while consecutive_failures < 10:  # pack the map
            if self._place_labyrinth(grid, clearance=1):
                consecutive_failures = 0
            else:
                consecutive_failures += 1

        return self._place_agents(grid)

    # ------------------------------------------------------------------ #
    # Single labyrinth generation / placement
    # ------------------------------------------------------------------ #
    def _place_labyrinth(self, grid: np.ndarray, clearance: int) -> bool:
        pattern = self._generate_labyrinth()
        return self._place_region(grid, pattern, clearance)

    def _generate_labyrinth(self) -> np.ndarray:
        # --- choose elongated size (smaller) -------------------------- #
        short = int(self._rng.integers(4, 8))  # 4–7
        long = int(self._rng.integers(short + 3, short + 8))  # +3–7
        if self._rng.random() < 0.5:  # horizontal
            h, w, entrance = short, long, (1, 0)
        else:  # vertical
            h, w, entrance = long, short, (0, 1)
        altar = (h - 2, w - 2)

        # keep dimensions odd for maze carving
        h += h % 2 == 0
        w += w % 2 == 0

        wind = float(self._rng.random())  # 0 = straight, 1 = twisty
        pat = np.full((h, w), "wall", dtype=object)
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        start = (1, 1)
        pat[start] = "empty"
        stack = [start]
        last_dir: Optional[Tuple[int, int]] = None

        while stack:
            r, c = stack[-1]
            options = [
                (dr, dc)
                for dr, dc in dirs
                if 0 < r + dr * 2 < h - 1 and 0 < c + dc * 2 < w - 1 and pat[r + dr * 2, c + dc * 2] == "wall"
            ]
            if options:
                if last_dir in options and self._rng.random() > wind:
                    dr, dc = last_dir
                else:
                    dr, dc = options[self._rng.integers(len(options))]
                pat[r + dr, c + dc] = "empty"
                pat[r + dr * 2, c + dc * 2] = "empty"
                stack.append((r + dr * 2, c + dc * 2))
                last_dir = (dr, dc)
            else:
                stack.pop()
                last_dir = None

        # carve entrance and place altar
        pat[entrance] = "empty"
        pat[altar] = "altar"
        return pat

    # ------------------------------------------------------------------ #
    # Agent placement (outside labyrinths)
    # ------------------------------------------------------------------ #
    def _place_agents(self, grid: np.ndarray):
        agent_tags = (
            ["agent.agent"] * self._agents
            if isinstance(self._agents, int)
            else ["agent." + t for t, n in self._agents.items() for _ in range(n)]
        )
        for tag in agent_tags:
            pos = self._random_empty()
            if pos:
                grid[pos] = tag
                self._occ[pos] = True
        return grid

    # ------------------------------------------------------------------ #
    # Region placement utilities
    # ------------------------------------------------------------------ #
    def _place_region(self, grid, pattern: np.ndarray, clearance: int) -> bool:
        ph, pw = pattern.shape
        for r, c in self._free_windows((ph + 2 * clearance, pw + 2 * clearance)):
            # stamp the pattern
            grid[r + clearance : r + clearance + ph, c + clearance : c + clearance + pw] = pattern
            # mark the *entire rectangle* as occupied so agents stay outside
            self._occ[r : r + ph + 2 * clearance, c : c + pw + 2 * clearance] = True
            return True
        return False

    def _free_windows(self, shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        h, w = shape
        H, W = self._occ.shape
        if h > H or w > W:
            return []
        view_shape = (H - h + 1, W - w + 1, h, w)
        sub = np.lib.stride_tricks.as_strided(self._occ, view_shape, self._occ.strides * 2)
        coords = np.argwhere(sub.sum(axis=(2, 3)) == 0)
        self._rng.shuffle(coords)
        return [tuple(t) for t in coords]

    def _random_empty(self) -> Optional[Tuple[int, int]]:
        empties = np.flatnonzero(~self._occ)
        if empties.size == 0:
            return None
        idx = self._rng.integers(empties.size)
        return tuple(np.unravel_index(empties[idx], self._occ.shape))
