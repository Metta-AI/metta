"""
NarrowWorld: multi-agent environment with 1-tile-wide snaking corridors.
Agents move through narrow channels (generated as a maze).
Single empty bays (1×1) are carved randomly to allow passing.
mine_reds, generators and altars are scattered along corridor cells in equal proportion.
"""

from typing import List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room


class NarrowWorld(Room):
    def __init__(
        self,
        width: int,
        height: int,
        objects: DictConfig,
        agents: int = 15,
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: str = "wall",
        pass_bay_prob: float = 0.05,
        team: str = "agent",
    ) -> None:
        super().__init__(border_width=border_width, border_object=border_object, labels=["narrow_world"])
        self.set_size_labels(width, height)
        self._w = width
        self._h = height
        self._agents_n = agents
        self._objects_cfg = objects
        self._rng = np.random.default_rng(seed)
        self._pass_bay_prob = pass_bay_prob
        self._team = team

    def _build(self) -> np.ndarray:  # type: ignore[override]
        grid = np.full((self._h, self._w), "wall", dtype=object)
        self._occ = np.ones((self._h, self._w), dtype=bool)
        self._carve_maze()
        self._add_passing_bays()
        # Reflect corridors in grid
        grid[~self._occ] = "empty"

        counts = {k: int(self._objects_cfg.get(k, 0)) for k in ("altar", "mine_red", "generator_red")}
        object_cycle = ["altar", "mine_red", "generator_red"]
        corridor_cells = [(int(r), int(c)) for r, c in zip(*np.where(~self._occ), strict=False)]
        self._rng.shuffle(corridor_cells)

        def place_adjacent(obj: str) -> bool:
            for _ in range(1000):  # attempt limit
                if not corridor_cells:
                    return False
                r, c = corridor_cells[self._rng.integers(0, len(corridor_cells))]
                neighs = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
                self._rng.shuffle(neighs)
                for nr, nc in neighs:
                    if 0 <= nr < self._h and 0 <= nc < self._w and self._occ[nr, nc]:
                        # convert wall to object spot if currently wall
                        grid[nr, nc] = obj
                        self._occ[nr, nc] = True
                        return True
            return False

        for obj_name in object_cycle:
            for _ in range(counts.get(obj_name, 0)):
                place_adjacent(obj_name)

        empty_cells = [(r, c) for r, c in corridor_cells if not self._occ[r, c]]
        self._rng.shuffle(empty_cells)
        for i in range(min(self._agents_n, len(empty_cells))):
            r, c = empty_cells[i]
            grid[r, c] = f"agent.{self._team}"
            self._occ[r, c] = True
        return grid

    # maze
    def _carve_maze(self):
        H, W = self._h, self._w
        stack: List[Tuple[int, int]] = []
        grid = np.zeros((H, W), dtype=int)
        start = (1, 1)
        grid[start] = 1
        stack.append(start)
        dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        while stack:
            r, c = stack[-1]
            nbrs = []
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 1 <= nr < H - 1 and 1 <= nc < W - 1 and grid[nr, nc] == 0:
                    nbrs.append((nr, nc, dr, dc))
            if nbrs:
                nr, nc, dr, dc = nbrs[self._rng.integers(0, len(nbrs))]
                grid[r + dr // 2, c + dc // 2] = 1
                grid[nr, nc] = 1
                stack.append((nr, nc))
            else:
                stack.pop()
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 1:
                    self._occ[r, c] = False

    def _add_passing_bays(self):
        for r in range(1, self._h - 1):
            for c in range(1, self._w - 1):
                if self._occ[r, c] and self._rng.random() < self._pass_bay_prob:
                    neighs = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
                    self._rng.shuffle(neighs)
                    for nr, nc in neighs:
                        if not self._occ[nr, nc]:
                            self._occ[r, c] = False
                            break
