"""BoxShare multi-agent environment.
A rectangular wall box is placed in the map.  Roughly half the agents spawn inside
that box, the others outside.  Some wall blocks of the box are replaced by
`generator` objects.  Inside the box several `mine` objects are scattered, while
`altar` objects are placed outside.  All counts are read from the YAML objects
DictConfig (defaults provided).
"""

from typing import List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room


class BoxShare(Room):
    def __init__(
        self,
        width: int,
        height: int,
        objects: DictConfig | None = None,
        agents: int = 14,
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: str = "wall",
        team: str = "agent",
    ) -> None:
        super().__init__(border_width=border_width, border_object=border_object, labels=["boxshare"])
        self.set_size_labels(width, height)
        self._w = width
        self._h = height
        self._agents_n = agents
        self._objects_cfg = objects or DictConfig({})
        self._rng = np.random.default_rng(seed)
        self._team = team

    def _to_int(self, val, default: int) -> int:
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    def _build(self) -> np.ndarray:  # type: ignore[override]
        grid = np.full((self._h, self._w), "empty", dtype=object)
        self._occ = np.zeros((self._h, self._w), dtype=bool)

        # ---- build wall box ----
        box_w = int(self._rng.integers(10, min(20, self._w - 4)))
        box_h = int(self._rng.integers(10, min(20, self._h - 4)))
        top = (self._h - box_h) // 2
        left = (self._w - box_w) // 2
        wall_coords: List[Tuple[int, int]] = []
        for r in range(top, top + box_h):
            for c in range(left, left + box_w):
                if r in (top, top + box_h - 1) or c in (left, left + box_w - 1):
                    grid[r, c] = "wall"
                    self._occ[r, c] = True
                    wall_coords.append((r, c))
        # ---- replace some walls with generators ----
        gen_count = self._to_int(self._objects_cfg.get("generator_red", 8), 8)
        self._rng.shuffle(wall_coords)
        for r, c in wall_coords[:gen_count]:
            grid[r, c] = "generator_red"
        # mark as occupied already

        # ---- place mines inside ----
        in_coords = [(r, c) for r in range(top + 1, top + box_h - 1) for c in range(left + 1, left + box_w - 1)]
        self._rng.shuffle(in_coords)
        mine_count = self._to_int(self._objects_cfg.get("mine_red", 7), 7)
        placed = 0
        for r, c in in_coords:
            if placed >= mine_count:
                break
            if grid[r, c] == "empty":
                grid[r, c] = "mine_red"
                self._occ[r, c] = True
                placed += 1

        # ---- place altars outside ----
        out_coords = [
            (r, c)
            for r in range(self._h)
            for c in range(self._w)
            if grid[r, c] == "empty" and not (left < c < left + box_w - 1 and top < r < top + box_h - 1)
        ]
        self._rng.shuffle(out_coords)
        altar_count = self._to_int(self._objects_cfg.get("altar", 7), 7)
        for r, c in out_coords[:altar_count]:
            grid[r, c] = "altar"
            self._occ[r, c] = True

        # ---- place agents ----
        inside_agents = self._agents_n // 2
        self._rng.shuffle(in_coords)
        for r, c in [p for p in in_coords if grid[p[0], p[1]] == "empty"][:inside_agents]:
            grid[r, c] = f"agent.{self._team}"
            self._occ[r, c] = True
        self._rng.shuffle(out_coords)
        for r, c in [p for p in out_coords if grid[p[0], p[1]] == "empty"][: (self._agents_n - inside_agents)]:
            grid[r, c] = f"agent.{self._team}"
            self._occ[r, c] = True

        return grid
