"""
SnakeyCylinder environment.

Generates a network of narrow tube-like corridors ("cylinders") that snake around the
map. Each corridor consists of two parallel walls enclosing a 1-tile-wide passage.
Corridor segments have random length 4-12 and randomly turn at right angles to create
long winding tubes.  At random points small alcoves (wall displacements) are carved so
agents can pass each other.  Entry points from outside are added by opening some wall
segments to the exterior.

Objects (altar, mine, generator) are placed in alcoves so they do not block the main
passage yet remain sealed from the outside world.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from mettagrid.room.room import Room


class SnakeyCylinder(Room):
    """Room generator for snakey cylinder network."""

    # Direction vectors (row, col)
    _DIRS: list[Tuple[int, int]] = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W

    def __init__(
        self,
        width: int,
        height: int,
        objects: DictConfig,
        agents: int | dict = 1,
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: str = "wall",
        teams: list | None = None,
        turn_prob: float = 0.25,
    ) -> None:
        super().__init__(border_width=border_width, border_object=border_object, labels=["snakey_cylinder"])
        self.set_size_labels(width, height)
        self._width = width
        self._height = height
        self._objects_cfg = objects
        self._agents = agents
        self._teams = teams
        self._rng = np.random.default_rng(seed)
        self._turn_prob = turn_prob

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------
    def _int_from_cfg(self, val) -> int:
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, DictConfig):
            try:
                return int(str(val))
            except (TypeError, ValueError):
                return 0
        if isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                return 0
        return 0

    def _object_counts(self) -> Dict[str, int]:
        return {k: self._int_from_cfg(self._objects_cfg.get(k, 0)) for k in ("altar", "mine", "generator")}

    # ------------------------------------------------------------------
    def _build(self) -> np.ndarray:  # noqa: C901
        # Start with solid walls
        grid = np.full((self._height, self._width), "wall", dtype=object)

        # Carve snake corridors
        walkway_cells: list[Tuple[int, int]] = []
        alcove_cells: list[Tuple[int, int]] = []

        # Choose random start internal point (keep 2-tile margin)
        r = int(self._rng.integers(2, self._height - 2))
        c = int(self._rng.integers(2, self._width - 2))
        dir_idx = int(self._rng.integers(0, 4))

        target_walk = int((self._width * self._height) * 0.3)  # 30% coverage
        steps = 0
        while steps < target_walk:
            length = int(self._rng.integers(4, 13))  # 4-12 inclusive
            dr, dc = self._DIRS[dir_idx]
            for _ in range(length):
                if not (1 <= r < self._height - 1 and 1 <= c < self._width - 1):
                    break  # hit border
                # Carve passage cell
                grid[r, c] = "empty"
                walkway_cells.append((r, c))

                # Maybe create an alcove (10% chance)
                if self._rng.random() < 0.1:
                    # Choose side: left (-1) or right (+1) relative to direction index
                    side_dir = (dir_idx - 1) % 4 if self._rng.random() < 0.5 else (dir_idx + 1) % 4
                    sr, sc = self._DIRS[side_dir]
                    ar, ac = r + sr, c + sc  # alcove cell (will become empty)
                    br, bc = ar + sr, ac + sc  # ensure exterior wall remains
                    if 0 <= ar < self._height and 0 <= ac < self._width and 0 <= br < self._height and 0 <= bc < self._width:
                        grid[ar, ac] = "empty"
                        # Leave wall at (br, bc) as sealing wall
                        alcove_cells.append((ar, ac))

                # Advance step
                r += dr
                c += dc
                steps += 1

            # Random turn
            if self._rng.random() < self._turn_prob:
                turn = -1 if self._rng.random() < 0.5 else 1  # left or right
                dir_idx = (dir_idx + turn) % 4

        # Create a few entry points from outside by opening border cells adjacent to passage
        border_entries = 0
        self._rng.shuffle(walkway_cells)
        for (wr, wc) in walkway_cells:
            if border_entries >= 4:
                break
            if wr == 1:
                grid[0, wc] = "empty"; border_entries += 1
            elif wr == self._height - 2:
                grid[self._height - 1, wc] = "empty"; border_entries += 1
            elif wc == 1:
                grid[wr, 0] = "empty"; border_entries += 1
            elif wc == self._width - 2:
                grid[wr, self._width - 1] = "empty"; border_entries += 1

        # Place objects in available alcoves (prefer) else in walkway cells
        counts = self._object_counts()
        place_cells = alcove_cells + walkway_cells
        self._rng.shuffle(place_cells)
        obj_symbols = []
        for obj_name, cnt in counts.items():
            obj_symbols.extend([obj_name] * cnt)
        self._rng.shuffle(obj_symbols)

        for symbol, (pr, pc) in zip(obj_symbols, place_cells):
            grid[pr, pc] = symbol

        # Place agents
        agent_syms: List[str] = []
        if self._teams is None:
            if isinstance(self._agents, int):
                agent_syms = ["agent.agent"] * self._agents
        else:
            if not isinstance(self._agents, int):
                raise ValueError("When using teams, 'agents' must be an int count")
            per_team = self._agents // len(self._teams) if self._teams else 0
            for t in self._teams:
                agent_syms += [f"agent.{t}"] * per_team

        empty_cells = [(r, c) for r, c in walkway_cells if grid[r, c] == "empty"]
        self._rng.shuffle(empty_cells)
        for sym in agent_syms:
            if not empty_cells:
                break
            pr, pc = empty_cells.pop()
            grid[pr, pc] = sym

        return grid


# Alias for config compatibility
SnakeyCylinderSequence = SnakeyCylinder
