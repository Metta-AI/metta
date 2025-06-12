from typing import Optional, Tuple

import numpy as np
from numba import njit

from mettagrid.room.room import Room


@njit
def _find_valid_positions_numba(occ: np.ndarray, pattern_height: int, pattern_width: int, clearance: int) -> np.ndarray:
    """Fast numba-compiled function to find valid positions for pattern placement."""
    H, W = occ.shape
    total_h = pattern_height + 2 * clearance
    total_w = pattern_width + 2 * clearance

    if H < total_h or W < total_w:
        return np.empty((0, 2), dtype=np.int32)

    valid_positions = []

    for r in range(H - total_h + 1):
        for c in range(W - total_w + 1):
            # Check if the region is clear
            is_clear = True
            for dr in range(total_h):
                for dc in range(total_w):
                    if occ[r + dr, c + dc]:
                        is_clear = False
                        break
                if not is_clear:
                    break

            if is_clear:
                valid_positions.append((r, c))

    return np.array(valid_positions, dtype=np.int32)


@njit
def _update_occupancy_numba(occ: np.ndarray, pattern_mask: np.ndarray, start_r: int, start_c: int, clearance: int):
    """Fast numba-compiled function to update occupancy grid."""
    ph, pw = pattern_mask.shape
    for dr in range(ph):
        for dc in range(pw):
            if pattern_mask[dr, dc]:
                occ[start_r + clearance + dr, start_c + clearance + dc] = True


@njit
def _get_empty_positions_numba(occ: np.ndarray) -> np.ndarray:
    """Fast numba-compiled function to get all empty positions."""
    H, W = occ.shape
    empty_positions = []

    for r in range(H):
        for c in range(W):
            if not occ[r, c]:
                empty_positions.append((r, c))

    return np.array(empty_positions, dtype=np.int32)


class CylinderWorld(Room):
    STYLE_PARAMETERS = {
        "cylinder_world": {
            "hearts_count": 0,  # altars are inside cylinders
            "cylinders": {"count": 999},  # ignored; we fill until no room
        },
    }

    def __init__(
        self,
        width: int,
        height: int,
        agents: int | dict = 0,
        seed: Optional[int] = 42,
        border_width: int = 0,
        border_object: str = "wall",
        team: str | None = None,
    ):
        super().__init__(border_width=border_width, border_object=border_object, labels=["cylinder_world"])
        self._rng = np.random.default_rng(seed)
        width, height = np.random.randint(40, 100), np.random.randint(40, 100)
        self._width, self._height = width, height
        self._agents = agents
        self._team = team
        # occupancy mask: False = empty
        self._occ = np.zeros((height, width), dtype=bool)

        # Pre-generate cylinder patterns for better performance
        self._pattern_cache = {}
        self._max_pattern_size = min(width, height) // 2

        self.set_size_labels(width, height)

    # ------------------------------------------------------------------ #
    # Public build
    # ------------------------------------------------------------------ #
    def _build(self) -> np.ndarray:
        return self._build_cylinder_world()

    # ------------------------------------------------------------------ #
    # Cylinder‑only build
    # ------------------------------------------------------------------ #
    def _build_cylinder_world(self) -> np.ndarray:
        """
        Optimized cylinder placement with early termination and better data structures.
        """
        # Use string array for better performance with fixed-size strings
        grid = np.full((self._height, self._width), "empty", dtype="U10")
        self._occ.fill(False)

        max_consecutive_fail = 2
        fails = 0
        placement_attempts = 0
        max_total_attempts = 1000  # Prevent infinite loops

        while fails < max_consecutive_fail and placement_attempts < max_total_attempts:
            placed = self._place_cylinder_once_optimized(grid, clearance=1)
            placement_attempts += 1

            if placed:
                fails = 0  # reset – we still found room
            else:
                fails += 1  # try a different size/orientation

        # Finally, spawn any requested agents on leftover empty cells
        grid = self._place_agents_optimized(grid)
        return grid

    # ------------------------------------------------------------------ #
    # Optimized cylinder placement helpers
    # ------------------------------------------------------------------ #
    def _place_cylinder_once_optimized(self, grid: np.ndarray, clearance: int = 1) -> bool:
        """Optimized cylinder placement with pattern caching."""
        pattern_data = self._generate_cylinder_pattern_optimized()
        if pattern_data is None:
            return False

        pattern, pattern_mask = pattern_data
        return self._place_region_optimized(grid, pattern, pattern_mask, clearance)

    def _generate_cylinder_pattern_optimized(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Generate cylinder pattern with caching and size limits."""
        max_length = min(30, self._max_pattern_size)
        if max_length < 2:
            return None

        length = int(self._rng.integers(2, max_length + 1))
        gap = int(self._rng.integers(1, 4))
        vertical = bool(self._rng.integers(0, 2))

        # Create cache key
        cache_key = (length, gap, vertical)
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]

        if vertical:
            h, w = length, gap + 2
            if h > self._height or w > self._width:
                return None
            pattern = np.full((h, w), "empty", dtype="U10")
            pattern_mask = np.zeros((h, w), dtype=bool)
            pattern[:, 0] = pattern[:, -1] = "wall"
            pattern_mask[:, 0] = pattern_mask[:, -1] = True
            altar_pos = (h // 2, 1 + gap // 2)
            pattern[altar_pos] = "altar"
            pattern_mask[altar_pos] = True
        else:
            h, w = gap + 2, length
            if h > self._height or w > self._width:
                return None
            pattern = np.full((h, w), "empty", dtype="U10")
            pattern_mask = np.zeros((h, w), dtype=bool)
            pattern[0, :] = pattern[-1, :] = "wall"
            pattern_mask[0, :] = pattern_mask[-1, :] = True
            altar_pos = (1 + gap // 2, w // 2)
            pattern[altar_pos] = "altar"
            pattern_mask[altar_pos] = True

        # Cache the result
        result = (pattern.copy(), pattern_mask.copy())
        if len(self._pattern_cache) < 100:  # Limit cache size
            self._pattern_cache[cache_key] = result

        return result

    # ------------------------------------------------------------------ #
    # Optimized agents placement
    # ------------------------------------------------------------------ #
    def _place_agents_optimized(self, grid):
        """Optimized agent placement using numba-compiled functions."""
        if self._agents == 0:
            return grid

        # Get all empty positions at once
        empty_positions = _get_empty_positions_numba(self._occ)

        if len(empty_positions) == 0:
            return grid

        # Determine agent strings
        if self._team is None:
            agent_str = "agent.agent"
        else:
            agent_str = f"agent.{self._team}"

        # Place agents
        num_to_place = min(self._agents, len(empty_positions))
        selected_indices = self._rng.choice(len(empty_positions), size=num_to_place, replace=False)

        for idx in selected_indices:
            r, c = empty_positions[idx]
            grid[r, c] = agent_str
            self._occ[r, c] = True

        return grid

    # ------------------------------------------------------------------ #
    # Optimized region placement utilities
    # ------------------------------------------------------------------ #
    def _place_region_optimized(
        self, grid: np.ndarray, pattern: np.ndarray, pattern_mask: np.ndarray, clearance: int
    ) -> bool:
        """Optimized region placement using numba-compiled functions."""
        ph, pw = pattern.shape

        # Find valid positions using numba
        valid_positions = _find_valid_positions_numba(self._occ, ph, pw, clearance)

        if len(valid_positions) == 0:
            return False

        # Randomly select one position
        selected_idx = self._rng.integers(0, len(valid_positions))
        r, c = valid_positions[selected_idx]

        # Place the pattern
        start_r, start_c = r + clearance, c + clearance
        grid[start_r : start_r + ph, start_c : start_c + pw] = pattern

        # Update occupancy using numba
        _update_occupancy_numba(self._occ, pattern_mask, r, c, clearance)

        return True

    def _rand_empty(self) -> Optional[Tuple[int, int]]:
        """Optimized random empty position selection."""
        empty_positions = _get_empty_positions_numba(self._occ)
        if len(empty_positions) == 0:
            return None

        idx = self._rng.integers(0, len(empty_positions))
        return tuple(empty_positions[idx])
