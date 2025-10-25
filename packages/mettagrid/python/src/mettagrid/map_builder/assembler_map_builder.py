from typing import Optional

import numpy as np

from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from mettagrid.map_builder.utils import draw_border


class AssemblerMapBuilderConfig(MapBuilderConfig["AssemblerMapBuilder"]):
    seed: Optional[int] = None

    width: int = 10
    height: int = 10
    objects: dict[str, int] = {}
    agents: int | dict[str, int] = 0
    border_width: int = 0
    border_object: str = "wall"

    # New: terrain density controller: "", "sparse", "balanced", "dense"
    terrain: str = "no-terrain"


class AssemblerMapBuilder(MapBuilder[AssemblerMapBuilderConfig]):
    def __init__(self, config):
        super().__init__(config)
        self._rng = np.random.default_rng(self.config.seed)
        self._shape_cache: dict[tuple[str, int], np.ndarray] = {}

    # ---------- obstacle shapes ----------
    def _shape_square(self, size: int = 2) -> np.ndarray:
        return np.full((size, size), "wall", dtype="<U50")

    def _shape_cross(self, size: int = 2) -> np.ndarray:
        s = size * 2 - 1
        out = np.full((s, s), "empty", dtype="<U50")
        mid = size - 1
        out[mid, :] = "wall"
        out[:, mid] = "wall"
        return out

    def _shape_L(self, size: int = 2) -> np.ndarray:
        out = np.full((size, size), "empty", dtype="<U50")
        out[:, 0] = "wall"
        out[size - 1, :] = "wall"
        return out

    def _get_shape(self, kind: str, size: int) -> np.ndarray:
        key = (kind, size)
        if key in self._shape_cache:
            return self._shape_cache[key]
        if kind == "square":
            arr = self._shape_square(size)
        elif kind == "cross":
            arr = self._shape_cross(size)
        elif kind == "L":
            arr = self._shape_L(size)
        elif kind == "block":
            arr = np.array([["wall"]], dtype="<U50")
        else:
            arr = np.array([["wall"]], dtype="<U50")
        self._shape_cache[key] = arr
        return arr

    def _choose_random_obstacle(self) -> np.ndarray:
        """
        Randomly pick a shape with a slight bias toward small, varied pieces.
        Balanced for diversity and packability.
        """
        # Probabilities tuned to avoid over-blocking tiny maps
        kinds = np.array(["block", "square", "L", "cross"])
        probs = np.array([0.40, 0.30, 0.20, 0.10])
        kind = self._rng.choice(kinds, p=probs)
        # size 2 is visually meaningful yet compact; blocks ignore size
        size = 2
        return self._get_shape(kind, size)

    def _get_num_obstacles(self, inner_area: int) -> int:
        t = getattr(self.config, "terrain", "no-terrain") or "no-terrain"
        if t == "sparse":
            return max(1, inner_area // 40)
        if t == "balanced":
            return max(2, inner_area // 22)
        if t == "dense":
            return max(3, inner_area // 14)
        return 0

    # ---------- small utils ----------
    @staticmethod
    def _dilate_bool(mask: np.ndarray, radius: int = 1) -> np.ndarray:
        """
        Fast Chebyshev dilation using boolean shifts (no scipy).
        """
        h, w = mask.shape
        out = np.zeros_like(mask, dtype=bool)
        for di in range(-radius, radius + 1):
            si0 = max(0, -di)
            ti0 = max(0, di)
            ih = h - abs(di)
            if ih <= 0:
                continue
            for dj in range(-radius, radius + 1):
                sj0 = max(0, -dj)
                tj0 = max(0, dj)
                jw = w - abs(dj)
                if jw <= 0:
                    continue
                out[ti0 : ti0 + ih, tj0 : tj0 + jw] |= mask[si0 : si0 + ih, sj0 : sj0 + jw]
        return out

    # ---------- main ----------
    def build(self):
        # Reset RNG for deterministic builds across calls
        if self.config.seed is not None:
            self._rng = np.random.default_rng(self.config.seed)

        H = self.config.height
        W = self.config.width
        bw = self.config.border_width

        # Empty grid
        grid = np.full((H, W), "empty", dtype="<U50")

        # Border
        if bw > 0:
            draw_border(grid, bw, self.config.border_object)

        # Inner dims
        inner_h = max(0, H - 2 * bw)
        inner_w = max(0, W - 2 * bw)
        if inner_h < 1 or inner_w < 1:
            return GameMap(grid)

        # ---------- Place terrain obstacles first ----------
        inner_area = inner_h * inner_w
        n_obs = self._get_num_obstacles(inner_area)

        if n_obs > 0:
            # To avoid expensive full candidate scans each time, do bounded random trials.
            # We also prevent obstacles from “spilling” outside the 1-cell halo area we
            # reserve for objects later, by placing obstacles within [bw : H - bw), but
            # we allow them up to the inner border; objects themselves will enforce a +1 margin.
            # For safe padding checks during placement, we’ll keep candidates inside [bw : H), etc.
            for _ in range(n_obs):
                shape = self._choose_random_obstacle()
                sh, sw = shape.shape
                # valid upper-left corners so that entire shape fits inside grid
                # (no extra halo required around terrain itself)
                i_min = bw
                j_min = bw
                i_max = H - bw - sh
                j_max = W - bw - sw
                if i_max < i_min or j_max < j_min:
                    break

                # Try up to K random placements
                K = 200
                placed = False
                for _try in range(K):
                    i = int(self._rng.integers(i_min, i_max + 1))
                    j = int(self._rng.integers(j_min, j_max + 1))
                    region = grid[i : i + sh, j : j + sw]
                    # place only on empty cells
                    if np.any(region != "empty"):
                        continue
                    # stamp walls where shape == wall
                    mask = shape == "wall"
                    if not mask.any():
                        continue
                    region[mask] = "wall"
                    placed = True
                    break
                if not placed:
                    # give up on this obstacle; continue to next
                    continue

        # ---------- Prepare object placement with halo constraints ----------
        # Objects must:
        # 1) be inside a 1-cell margin from all borders (so their 3x3 neighborhood is in-bounds),
        # 2) have their 3x3 neighborhood entirely empty,
        # 3) not be adjacent to terrain (wall) — enforced by dilating walls and forbidding those cells.

        # forbid cells on/next-to walls
        walls = grid == "wall"
        forbidden = self._dilate_bool(walls, radius=1)  # walls + Moore-1 neighborhood

        # Keep track of object halos in same mask
        blocked = forbidden.copy()

        # bounds that keep the 3x3 fully in-bounds
        top = bw + 1
        left = bw + 1
        bottom = H - bw - 2
        right = W - bw - 2
        if bottom < top or right < left:
            # No room to place any objects with required halo; just place agents & return
            self._place_agents(grid)
            return GameMap(grid)

        # Precompute candidate coordinates once, shuffled
        cand_rows = np.arange(top, bottom + 1)
        cand_cols = np.arange(left, right + 1)
        candidates = np.stack(np.meshgrid(cand_rows, cand_cols, indexing="ij"), axis=-1).reshape(-1, 2)
        self._rng.shuffle(candidates)

        # Flatten object symbols list
        object_symbols: list[str] = []
        for name, count in self.config.objects.items():
            if count > 0:
                object_symbols.extend([name] * count)

        # Efficient local checks via slicing
        def is_clear_3x3(i: int, j: int) -> bool:
            # center empty and whole 3x3 empty & unblocked
            if grid[i, j] != "empty":
                return False
            i0, i1 = i - 1, i + 2
            j0, j1 = j - 1, j + 2
            # both conditions at once minimizes cache misses
            region_block = blocked[i0:i1, j0:j1]
            region_grid = grid[i0:i1, j0:j1]
            return (not region_block.any()) and (region_grid == "empty").all()

        def reserve_halo(i: int, j: int):
            blocked[i - 1 : i + 2, j - 1 : j + 2] = True

        if object_symbols and len(candidates) > 0:
            # Greedy placement honoring constraints; O(N) over candidates
            idx = 0
            for symbol in object_symbols:
                placed = False
                # advance through candidates until we find a legal slot
                while idx < len(candidates):
                    i, j = int(candidates[idx][0]), int(candidates[idx][1])
                    idx += 1
                    if is_clear_3x3(i, j):
                        grid[i, j] = symbol
                        reserve_halo(i, j)
                        placed = True
                        break
                if not placed:
                    # no more valid slots under constraints
                    break

        # ---------- Place agents (no special padding) ----------
        self._place_agents(grid)

        return GameMap(grid)

    # ---------- helpers ----------
    def _place_agents(self, grid: np.ndarray) -> None:
        if isinstance(self.config.agents, int):
            agent_symbols = ["agent.agent"] * self.config.agents
        elif isinstance(self.config.agents, dict):
            agent_symbols = ["agent." + a for a, n in self.config.agents.items() for _ in range(n)]
        else:
            raise ValueError(f"Invalid agents configuration: {self.config.agents}")

        if not agent_symbols:
            return

        empties = np.argwhere(grid == "empty")
        if len(empties) == 0:
            return
        self._rng.shuffle(empties)
        k = min(len(agent_symbols), len(empties))
        for t in range(k):
            i, j = int(empties[t][0]), int(empties[t][1])
            grid[i, j] = agent_symbols[t]
