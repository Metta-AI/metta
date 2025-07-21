"""
Combined-Adversarial Terrain
============================
A single room containing (in *one* layout) every feature that breaks the
classic memory-less strategies listed in your design notes:

* disconnected, jagged **islands** → wall-followers loop forever ext.rtf](file-service://file-LmBR3Z8mT3t4MaDuyQLgwH)
* huge open **plaza** + deep cul-de-sacs → slow pure random walk coverage ext.rtf](file-service://file-LmBR3Z8mT3t4MaDuyQLgwH)
* ±90° **parallel-corridor labyrinth** + U-bays → traps direction-biased walkers ext.rtf](file-service://file-LmBR3Z8mT3t4MaDuyQLgwH)
* concave **C-pockets** & **fly-traps** → bug-0 bounce-and-turn loops ext.rtf](file-service://file-LmBR3Z8mT3t4MaDuyQLgwH)
* **saw-tooth perimeter** & interior blocks → lawn-mower rows mis-align xt.rtf](file-service://file-LmBR3Z8mT3t4MaDuyQLgwH)
* an early off-centre obstacle maze → outward spirals stall xt.rtf](file-service://file-LmBR3Z8mT3t4MaDuyQLgwH)
* irregular gaps in the corridor grid → universal turn sequences lose their guarantee xt.rtf](file-service://file-LmBR3Z8mT3t4MaDuyQLgwH)

All sub-structures are parameterised so you can sweep them from YAML.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room


class CombinedAdversarialTerrain(Room):
    """
    Kitchen-sink terrain that simultaneously frustrates *every* zero-memory rule.
    """

    # --------------------------------------------------------------------- #
    # Construction                                                          #
    # --------------------------------------------------------------------- #
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
        # ─ islands (wall-follow trap)
        island_count: int = 60,
        island_size: int = 18,
        #
        # ─ cul-de-sacs (pure random-walk trap)
        corridor_count: int = 40,
        corridor_length: int = 45,
        #
        # ─ labyrinth & U-bays (direction-bias trap)
        hole_count_range: int = 5,
        u_bay_count: int = 40,
        u_bay_depth: int = 17,
        u_bay_width: int = 7,
        #
        # ─ concave pockets / fly-traps (bug-0 trap)
        pocket_count: int = 30,
        pocket_size: int = 10,
        trap_count: int = 30,
        trap_depth: int = 15,
        trap_width: int = 8,
        #
        # ─ jagged perimeter (lawn-mower trap)
        indent_count: int = 22,
        indent_depth: int = 6,
        indent_width: int = 5,
        #
        corridor_spacing: int = 5,
        occupancy_threshold: float = 0.65,
    ) -> None:
        super().__init__(
            border_width=border_width,
            border_object=border_object,
            labels=["combined_adversarial"],
        )
        self.set_size_labels(width, height)

        self._H, self._W = height, width
        self._rng = np.random.default_rng(seed)
        self._agents = agents if isinstance(agents, int) else 1
        self._objects = objects or {}

        # params
        self._island_cnt = int(island_count)
        self._island_sz = int(island_size)
        self._corridor_cnt = int(corridor_count)
        self._corr_len = int(corridor_length)
        # gap (in tiles) between successive corridor walls (controls spaciousness)
        self._corridor_spacing = int(corridor_spacing)
        self._hole_rng = int(hole_count_range)
        self._u_cnt = int(u_bay_count)
        self._u_dep = int(u_bay_depth)
        self._u_wid = int(u_bay_width)
        self._pocket_cnt = int(pocket_count)
        self._pocket_sz = int(pocket_size)
        self._trap_cnt = int(trap_count)
        self._trap_dep = int(trap_depth)
        self._trap_wid = int(trap_width)
        self._indent_cnt = int(indent_count)
        self._indent_dep = int(indent_depth)
        self._indent_wid = int(indent_width)
        self._occ_thr = occupancy_threshold

        self._occ = np.zeros((height, width), dtype=bool)

    # --------------------------------------------------------------------- #
    # Room-building pipeline                                                #
    # --------------------------------------------------------------------- #
    def _build(self) -> np.ndarray:
        grid = np.full((self._H, self._W), "empty", dtype=object)

        # 0 ─ jagged perimeter
        self._carve_jagged_perimeter(grid)

        # 1 ─ islands
        self._scatter_islands(grid)

        # 2 ─ parallel-corridor labyrinth (orientation 50 %)
        if self._rng.random() < 0.5:
            self._make_vertical_labyrinth(grid)
        else:
            self._make_horizontal_labyrinth(grid)

        # 3 ─ attach U-bays
        self._attach_u_bays(grid)

        # 4 ─ deep cul-de-sacs
        self._dig_culdesacs(grid)

        # 5 ─ concave pockets + fly-traps
        self._place_pockets_and_traps(grid)

        # 6 ─ agents
        for _ in range(self._agents):
            pos = self._pick_empty()
            if pos is None:
                break
            r, c = pos
            grid[r, c] = "agent.agent"
            self._occ[r, c] = True

        # 7 ─ user objects
        for name, cnt in self._objects.items():
            for _ in range(int(cnt)):
                pos = self._pick_empty()
                if pos is None:
                    break
                r, c = pos
                grid[r, c] = name
                self._occ[r, c] = True

        # 8 ─ ensure corridors from every agent to the nearest altar
        self._ensure_paths(grid)

        return grid

    # ------------------------------------------------------------------ #
    # 1 ─ Jagged perimeter                                               #
    # ------------------------------------------------------------------ #
    def _carve_jagged_perimeter(self, grid: np.ndarray) -> None:
        # continuous border first
        grid[0, :] = grid[-1, :] = "wall"
        grid[:, 0] = grid[:, -1] = "wall"
        self._occ[0, :], self._occ[-1, :] = True, True
        self._occ[:, 0], self._occ[:, -1] = True, True

        n_per_side = self._indent_cnt
        for side in ("top", "bottom", "left", "right"):
            for _ in range(n_per_side):
                depth = self._indent_dep
                width = self._indent_wid
                if side in ("top", "bottom"):
                    max_start = self._W - width - 2
                    if max_start <= 1:
                        continue
                    c0 = self._rng.integers(1, max_start)
                    rows = range(depth) if side == "top" else range(self._H - depth, self._H)
                    grid[np.array(rows)[:, None], c0 : c0 + width] = "wall"
                    self._occ[np.array(rows)[:, None], c0 : c0 + width] = True
                else:
                    max_start = self._H - width - 2
                    if max_start <= 1:
                        continue
                    r0 = self._rng.integers(1, max_start)
                    cols = range(depth) if side == "left" else range(self._W - depth, self._W)
                    grid[r0 : r0 + width, np.array(cols)] = "wall"
                    self._occ[r0 : r0 + width, np.array(cols)] = True

    # ------------------------------------------------------------------ #
    # 2 ─ Islands                                                        #
    # ------------------------------------------------------------------ #
    def _scatter_islands(self, grid: np.ndarray) -> None:
        n_islands = self._island_cnt
        for _ in range(n_islands):
            if self._occ.mean() >= self._occ_thr:
                break
            target = self._island_sz
            blob = self._make_jagged_blob(target)
            self._try_place(grid, blob, clearance=1)

    def _make_jagged_blob(self, target_cells: int) -> np.ndarray:
        cells = {(0, 0)}
        while len(cells) < target_cells:
            r, c = list(cells)[self._rng.integers(len(cells))]
            dr, dc = self._rng.choice([-1, 0, 1], size=2, replace=False)
            if dr == dc == 0:
                continue
            cells.add((r + dr, c + dc))
        # convert to tight array
        rs, cs = zip(*cells, strict=False)
        h, w = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
        arr = np.full((h, w), "wall", dtype=object)
        for r, c in cells:
            arr[r - min(rs), c - min(cs)] = "wall"
        return arr

    # ------------------------------------------------------------------ #
    # 3 ─ Parallel corridors + 4 ─ U-bays                                #
    # ------------------------------------------------------------------ #
    def _make_vertical_labyrinth(self, grid: np.ndarray) -> None:
        for c in range(2, self._W - 2, 2):
            grid[:, c] = "wall"
            self._occ[:, c] = True
        n_holes = self._hole_rng
        for _ in range(n_holes):
            c = int(self._rng.choice(range(2, self._W - 2, 2)))
            r0 = int(self._rng.integers(1, self._H - 2))
            grid[r0, c] = "empty"
            self._occ[r0, c] = False

    def _make_horizontal_labyrinth(self, grid: np.ndarray) -> None:
        for r in range(2, self._H - 2, 2):
            grid[r, :] = "wall"
            self._occ[r, :] = True
        n_holes = self._hole_rng
        for _ in range(n_holes):
            r = int(self._rng.choice(range(2, self._H - 2, 2)))
            c0 = int(self._rng.integers(1, self._W - 2))
            grid[r, c0] = "empty"
            self._occ[r, c0] = False

    def _attach_u_bays(self, grid: np.ndarray) -> None:
        for _ in range(self._u_cnt):
            depth = self._u_dep
            width = self._u_wid
            entry_row = int(self._rng.integers(1, self._H - depth - 2))
            entry_col = int(self._rng.integers(1, self._W - width - 2))
            # carve U
            for dc in range(width + 1):
                grid[entry_row, entry_col + dc] = "empty"
            for dr in range(1, depth + 1):
                grid[entry_row + dr, entry_col + 0] = "empty"
                grid[entry_row + dr, entry_col + width] = "empty"
            for dc in range(1, width):
                grid[entry_row + depth, entry_col + dc] = "empty"
            self._occ[grid == "empty"] = False  # refresh mask lazily

    # ------------------------------------------------------------------ #
    # 5 ─ Cul-de-sacs                                                    #
    # ------------------------------------------------------------------ #
    def _dig_culdesacs(self, grid: np.ndarray) -> None:
        for _ in range(self._corridor_cnt):
            L = self._corr_len
            if self._rng.random() < 0.5:  # horizontal corridor
                r = int(self._rng.integers(1, self._H - 1))
                c0 = int(self._rng.integers(1, self._W - L - 1))
                grid[r, c0 : c0 + L] = "empty"
            else:  # vertical corridor
                c = int(self._rng.integers(1, self._W - 1))
                r0 = int(self._rng.integers(1, self._H - L - 1))
                grid[r0 : r0 + L, c] = "empty"
        self._occ[grid == "empty"] = False

    # ------------------------------------------------------------------ #
    # 6 ─ Pockets & fly-traps                                            #
    # ------------------------------------------------------------------ #
    def _place_pockets_and_traps(self, grid: np.ndarray) -> None:
        for _ in range(self._pocket_cnt):
            size = self._pocket_sz
            r = int(self._rng.integers(2, self._H - size - 2))
            c = int(self._rng.integers(2, self._W - size - 2))
            # carve C-pocket facing east
            grid[r : r + size, c] = "wall"
            grid[r, c : c + size] = "wall"
            grid[r + size - 1, c : c + size] = "wall"
            self._occ[r : r + size, c] = True
            self._occ[r, c : c + size] = True
            self._occ[r + size - 1, c : c + size] = True

        for _ in range(self._trap_cnt):
            depth = self._trap_dep
            width = self._trap_wid | 1  # odd
            r = int(self._rng.integers(2, self._H - depth - 2))
            c = int(self._rng.integers(2, self._W - width - 2))
            mid = c + width // 2
            # two columns form funnel
            grid[r : r + depth, mid - 1] = "wall"
            grid[r : r + depth, mid + 1] = "wall"
            grid[r + depth - 1, mid - 1 : mid + 2] = "wall"
            self._occ[r : r + depth, mid - 1] = True
            self._occ[r : r + depth, mid + 1] = True
            self._occ[r + depth - 1, mid - 1 : mid + 2] = True

    # ------------------------------------------------------------------ #
    # 8 ─ Connectivity helpers                                           #
    # ------------------------------------------------------------------ #
    def _ensure_paths(self, grid: np.ndarray) -> None:
        """
        After all agents and altars are placed, carve straight corridors
        (vertical then horizontal) so each agent has a clear path to at
        least one altar.
        """
        agent_pos = list(zip(*np.where(grid == "agent.agent"), strict=False))
        altar_pos = list(zip(*np.where(grid == "altar"), strict=False))
        if not agent_pos or not altar_pos:
            return

        for a_r, a_c in agent_pos:
            # choose the altar with minimum Manhattan distance
            g_r, g_c = min(altar_pos, key=lambda p: abs(p[0] - a_r) + abs(p[1] - a_c))
            self._carve_corridor(grid, (a_r, a_c), (g_r, g_c))

    def _carve_corridor(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> None:
        """Carve a two‑segment Manhattan corridor."""
        (r1, c1), (r2, c2) = start, goal

        # vertical leg
        step_r = 1 if r2 >= r1 else -1
        for r in range(r1, r2 + step_r, step_r):
            if grid[r, c1] not in ("agent.agent", "altar"):
                grid[r, c1] = "empty"
                self._occ[r, c1] = False

        # horizontal leg
        step_c = 1 if c2 >= c1 else -1
        for c in range(c1, c2 + step_c, step_c):
            if grid[r2, c] not in ("agent.agent", "altar"):
                grid[r2, c] = "empty"
                self._occ[r2, c] = False

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _try_place(self, grid: np.ndarray, pattern: np.ndarray, *, clearance: int) -> bool:
        ph, pw = pattern.shape
        H, W = self._H, self._W
        if ph + 2 * clearance > H or pw + 2 * clearance > W:
            return False
        cand: List[Tuple[int, int]] = []
        for r in range(clearance, H - ph - clearance):
            for c in range(clearance, W - pw - clearance):
                sub = self._occ[r - clearance : r + ph + clearance, c - clearance : c + pw + clearance]
                if not sub.any():
                    cand.append((r, c))
        if not cand:
            return False
        r, c = cand[self._rng.integers(len(cand))]
        grid[r : r + ph, c : c + pw] = pattern
        self._occ[r - clearance : r + ph + clearance, c - clearance : c + pw + clearance] = True
        return True

    def _pick_empty(self) -> Optional[Tuple[int, int]]:
        empty_flat = np.flatnonzero(~self._occ)
        if empty_flat.size == 0:
            return None
        idx = self._rng.integers(empty_flat.size)
        return np.unravel_index(idx, self._occ.shape)
