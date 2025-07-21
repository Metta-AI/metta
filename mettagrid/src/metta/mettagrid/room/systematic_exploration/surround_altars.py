"""
Surround-Altars Terrain
======================
Each altar is boxed by a wall ring with ONE doorway.

• Wall ring size can be sampled from a range (default 1 = 3×3 ring)
• Rings never overlap and keep appropriate gaps from the map border.
• Exactly `num_agents` agents are spawned (first on safe tiles, then fallback).
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
from omegaconf import DictConfig, ListConfig
from metta.mettagrid.room.room import Room

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E




class SurroundAltarsTerrain(Room):
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
        num_altars: int = 25,
        wall_size: int = 1,
        wall_gap_probability: float = 0.2,
    ) -> None:
        super().__init__(border_width=border_width,
                         border_object=border_object,
                         labels=["surround_altars"])
        self.set_size_labels(width, height)

        self.H, self.W = height, width
        self.rng = np.random.default_rng(seed)
        self.n_agents = int(agents)
        self.user_objects = objects if isinstance(objects, dict) else {}
        self.n_altars = int(num_altars)
        self.wall_size = int(wall_size)
        self.wall_gap_probability = wall_gap_probability

        self.occ = np.zeros((height, width), dtype=bool)

    # ------------------------------------------------------------------ #
    # Build                                                               #
    # ------------------------------------------------------------------ #
    def _build(self) -> np.ndarray:
        grid = np.full((self.H, self.W), "empty", dtype=object)

        # border walls
        grid[0, :], grid[-1, :] = "wall", "wall"
        grid[:, 0], grid[:, -1] = "wall", "wall"
        self.occ[0, :], self.occ[-1, :] = True, True
        self.occ[:, 0], self.occ[:, -1] = True, True

        self._place_guarded_altars(grid)
        self._place_agents(grid)
        self._place_extras(grid)

        return grid

    # ------------------------------------------------------------------ #
    # Altars                                                              #
    # ------------------------------------------------------------------ #
    def _place_guarded_altars(self, grid: np.ndarray) -> None:
        placed, attempts = 0, 0
        while placed < self.n_altars and attempts < self.n_altars * 200:
            attempts += 1
            wall_size = self.wall_size
            margin = wall_size + 1
            r = int(self.rng.integers(margin, self.H - margin))
            c = int(self.rng.integers(margin, self.W - margin))
            
            # Check if area is clear (altar + wall ring)
            if self.occ[r - wall_size : r + wall_size + 1, c - wall_size : c + wall_size + 1].any():
                continue
                
            doorway_dir = DIRS[int(self.rng.integers(len(DIRS)))]  # ensures tuple
            
            # centre altar
            grid[r, c] = "altar"
            self.occ[r, c] = True
            
            # wall ring
            for dr in range(-wall_size, wall_size + 1):
                for dc in range(-wall_size, wall_size + 1):
                    if dr == dc == 0:
                        continue  # skip center (altar)
                    # Only place walls on the perimeter of the ring
                    if abs(dr) == wall_size or abs(dc) == wall_size:
                        if (dr, dc) == doorway_dir:
                            continue  # leave doorway open
                        # Add random gaps with specified probability
                        if self.rng.random() < self.wall_gap_probability:
                            continue  # create gap by not placing wall
                        rr, cc = r + dr, c + dc
                        grid[rr, cc] = "wall"
                        self.occ[rr, cc] = True
            placed += 1

    # ------------------------------------------------------------------ #
    # Agents                                                              #
    # ------------------------------------------------------------------ #
    def _place_agents(self, grid: np.ndarray) -> None:
        rem = self.n_agents

        def safe_tile():
            empty = ~self.occ
            safe = empty.copy()
            for dr, dc in DIRS:
                safe &= ~np.roll(self.occ, shift=(dr, dc), axis=(0, 1))
            safe[0, :] = safe[-1, :] = False
            safe[:, 0] = safe[:, -1] = False
            flat = np.flatnonzero(safe)
            if flat.size:
                pos = np.unravel_index(self.rng.integers(flat.size), safe.shape)
                return pos
            return None

        while rem:
            pos = safe_tile()
            if pos is None:
                break
            grid[pos] = "agent.agent"
            self.occ[pos] = True
            rem -= 1

        while rem:
            flat = np.flatnonzero(~self.occ)
            if flat.size == 0:
                break
            pos = np.unravel_index(self.rng.integers(flat.size), self.occ.shape)
            grid[pos] = "agent.agent"
            self.occ[pos] = True
            rem -= 1

        if rem:
            raise RuntimeError(f"Unable to place {self.n_agents} agents (remaining {rem}).")

    # ------------------------------------------------------------------ #
    # Extras                                                              #
    # ------------------------------------------------------------------ #
    def _place_extras(self, grid: np.ndarray) -> None:
        for name, cnt in self.user_objects.items():
            for _ in range(int(cnt)):
                flat = np.flatnonzero(~self.occ)
                if flat.size == 0:
                    break
                pos = np.unravel_index(self.rng.integers(flat.size), self.occ.shape)
                grid[pos] = name
                self.occ[pos] = True
