"""
CylinderWorldSequence environment.
Generates randomly scattered narrow corridor cylinders (parallel wall pairs, 1-tile wide corridor).
Each cylinder has a random length between 4 and 12 tiles (inclusive).
Objects (altar, mine, generator) are placed at the center of each cylinder's corridor.
No objects are placed outside of these cylinders.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from mettagrid.room.room import Room


class CylinderWorldSequence(Room):
    """Grid room with randomly scattered cylindrical wall patterns (cylinders).

    Each cylinder consists of two parallel walls of equal length (sampled 4-12 inclusive)
    with a 1-tile wide corridor between them. An object (altar, mine, or generator)
    is placed at the exact center of this corridor.
    """

    DEFAULT_CYLINDER_PATTERN_WALL = np.array(
        [
            ["wall", "wall", "wall"],
            ["wall", "empty", "wall"],
            ["wall", "wall", "wall"],
        ],
        dtype=object,
    )

    def __init__(
        self,
        width: int,
        height: int,
        objects: DictConfig,  # expects counts for altar, mine, generator under room.objects
        agents: int = 1,
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: str = "wall",
        team: str = "agent",
        cylinder_pattern_type: str = "3x3wall",
    ) -> None:
        super().__init__(border_width=border_width, border_object=border_object, labels=["cylinder_sequence"])
        self.set_size_labels(width, height)
        self._rng = np.random.default_rng(seed)
        self._width = width
        self._height = height
        self._agents = agents
        self._team = team
        self._objects_cfg = objects

        self._cylinder_pattern = (
            self.DEFAULT_CYLINDER_PATTERN_WALL
            if cylinder_pattern_type == "3x3wall"
            else self.DEFAULT_CYLINDER_PATTERN_WALL
        )
        self._cyl_h, self._cyl_w = self._cylinder_pattern.shape

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _int_from_cfg(self, val) -> int:
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, DictConfig):
            try:
                return int(str(val))  # DictConfig might behave like string
            except (TypeError, ValueError):
                return 0
        if isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                return 0
        return 0

    def _object_counts(self) -> Dict[str, int]:
        return {k: self._int_from_cfg(self._objects_cfg.get(k, 0)) for k in ("altar", "mine.red", "generator.red")}

    # ------------------------------------------------------------------
    # Grid build
    # ------------------------------------------------------------------
    def _build(self) -> np.ndarray:  # noqa: C901
        """Build a room populated only with cylindrical corridors containing objects."""
        # Prepare agent symbols
        agent_syms = [f"agent.{self._team}"] * self._agents

        grid = np.full((self._height, self._width), "empty", dtype=object)
        self._occupancy = np.zeros((self._height, self._width), dtype=bool)

        # Build list of objects to place
        counts = self._object_counts()
        objects_pool: List[str] = []
        for name, cnt in counts.items():
            objects_pool.extend([name] * cnt)

        if objects_pool:
            objects_pool = list(self._rng.permutation(objects_pool))

        # Try to place cylinders until either pool exhausted or too many consecutive failures
        fail_streak, max_fail = 0, 50
        while objects_pool and fail_streak < max_fail:
            obj_name = objects_pool.pop()
            pattern = self._generate_cylinder_pattern(obj_name)
            if self._place_pattern(grid, pattern, clearance=1):
                fail_streak = 0
            else:
                fail_streak += 1
                # Requeue object for another attempt later
                objects_pool.insert(0, obj_name)

        # Place agents in randomly chosen empty cells across the whole grid
        empty_coords = list(zip(*np.where(grid == "empty"), strict=False))
        self._rng.shuffle(empty_coords)

        if len(empty_coords) < len(agent_syms):
            raise ValueError(
                f"Not enough empty cells ({len(empty_coords)}) to place {len(agent_syms)} agents. "
                "Increase map size or reduce agent count."
            )

        for sym, (pr, pc) in zip(agent_syms, empty_coords, strict=False):
            grid[pr, pc] = sym
            self._occupancy[pr, pc] = True

        return grid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_occ(self, top_left: Tuple[int, int], pattern: np.ndarray) -> None:
        r0, c0 = top_left
        h, w = pattern.shape
        # Mark only the walls/objects as occupied; corridor cells remain free for agents/objects
        self._occupancy[r0 : r0 + h, c0 : c0 + w] |= pattern != "empty"

    def _find_empty_regions(self, region_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        rh, rw = region_shape
        H, W = self._occupancy.shape
        if H < rh or W < rw:
            return []
        shape = (H - rh + 1, W - rw + 1, rh, rw)
        strides = self._occupancy.strides * 2
        windows = np.lib.stride_tricks.as_strided(self._occupancy, shape=shape, strides=strides)
        sums = windows.sum(axis=(2, 3))
        coords = np.argwhere(sums == 0)
        return [(int(x[0]), int(x[1])) for x in coords]

    def _choose_random_empty(self) -> Optional[Tuple[int, int]]:
        empty = np.flatnonzero(~self._occupancy)
        if empty.size == 0:
            return None
        idx = int(self._rng.integers(0, empty.size))
        unr = np.unravel_index(idx, self._occupancy.shape)
        return int(unr[0]), int(unr[1])

    # ------------------------------------------------------------------
    # Cylinder pattern and placement helpers
    # ------------------------------------------------------------------
    def _generate_cylinder_pattern(self, obj_name: str) -> np.ndarray:
        """Create a cylinder pattern: two parallel walls with a 1-tile corridor and an object at its center.

        The cylinder length is randomly sampled between 4 and 12 tiles, inclusive.
        The cylinder can be oriented vertically or horizontally.
        """
        # Cylinder length is sampled to be between 4 and 12 (inclusive).
        length = int(self._rng.integers(4, 13))  # high=13 is exclusive, so 4-12.
        vertical = bool(self._rng.integers(0, 2))
        if vertical:
            # Vertical corridor: two vertical walls separated by one-tile gap
            h, w = length, 3  # width fixed to 3 (wall, path, wall)
            pat = np.full((h, w), "empty", dtype=object)
            pat[:, 0] = pat[:, 2] = "wall"
            pat[h // 2, 1] = obj_name  # place object at centre of corridor
        else:
            # Horizontal corridor: two horizontal walls separated by one-tile gap
            h, w = 3, length  # height fixed to 3
            pat = np.full((h, w), "empty", dtype=object)
            pat[0, :] = pat[2, :] = "wall"
            pat[1, w // 2] = obj_name
        return pat

    def _place_pattern(self, grid: np.ndarray, pattern: np.ndarray, clearance: int = 1) -> bool:
        ph, pw = pattern.shape
        candidates = self._find_empty_regions((ph + 2 * clearance, pw + 2 * clearance))
        if not candidates:
            return False
        r0, c0 = candidates[self._rng.integers(0, len(candidates))]
        grid[r0 + clearance : r0 + clearance + ph, c0 + clearance : c0 + clearance + pw] = pattern
        self._update_occ((r0 + clearance, c0 + clearance), pattern)
        return True


# Backwards compatibility alias
CylinderObjectSequence = CylinderWorldSequence
