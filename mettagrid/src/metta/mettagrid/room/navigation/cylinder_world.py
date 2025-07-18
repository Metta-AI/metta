import random
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import convolve2d

from metta.mettagrid.room.room import Room


class CylinderWorld(Room):
    def __init__(
        self,
        width: int,
        height: int,
        agents: int | dict = 0,
        seed: Optional[int] = 42,
        border_width: int = 0,
        border_object: str = "wall",
        team: str | None = None,
        agent_cluster_type: str = "no_clustering",
    ):
        super().__init__(border_width=border_width, border_object=border_object, labels=["cylinder_world"])
        self._rng = np.random.default_rng(seed)
        width, height = np.random.randint(40, 100), np.random.randint(40, 100)
        self._width, self._height = width, height
        self._agents = agents
        self._team = team
        # occupancy mask: False = empty
        self._occ = np.zeros((height, width), dtype=bool)
        self._agent_cluster_type = agent_cluster_type

        self.set_size_labels(width, height)

    def get_valid_positions(self, level):
        # Create a boolean mask for empty cells
        empty_mask = level == "empty"

        # Use numpy's roll to check adjacent cells efficiently
        has_empty_neighbor = (
            np.roll(empty_mask, 1, axis=0)  # Check up
            | np.roll(empty_mask, -1, axis=0)  # Check down
            | np.roll(empty_mask, 1, axis=1)  # Check left
            | np.roll(empty_mask, -1, axis=1)  # Check right
        )

        # Valid positions are empty cells with at least one empty neighbor
        # Exclude border cells (indices 0 and -1)
        valid_mask = empty_mask & has_empty_neighbor
        valid_mask[0, :] = False
        valid_mask[-1, :] = False
        valid_mask[:, 0] = False
        valid_mask[:, -1] = False

        # Get coordinates of valid positions
        valid_positions = list(zip(*np.where(valid_mask), strict=False))
        return valid_positions

    def right_next_to_each_other_positions(self, level, num_agents):
        # Create a boolean mask for empty cells
        empty_mask = level == "empty"

        # Use numpy's roll to check adjacent cells efficiently
        has_empty_neighbor = (
            np.roll(empty_mask, 1, axis=0)  # Check up
            | np.roll(empty_mask, -1, axis=0)  # Check down
            | np.roll(empty_mask, 1, axis=1)  # Check left
            | np.roll(empty_mask, -1, axis=1)  # Check right
        )

        # Valid positions are empty cells with at least one empty neighbor
        # Exclude border cells (indices 0 and -1)
        valid_mask = empty_mask & has_empty_neighbor
        valid_mask[0, :] = False
        valid_mask[-1, :] = False
        valid_mask[:, 0] = False
        valid_mask[:, -1] = False
        # Find all valid positions (where valid_mask is True)
        valid_indices = np.argwhere(valid_mask)
        if len(valid_indices) == 0:
            return []

        # Pick a random starting index
        start_idx = valid_indices[np.random.choice(len(valid_indices))]
        start_pos = tuple(start_idx)

        # Flood-fill to find num_agents contiguous valid positions
        from collections import deque

        visited = set()
        contiguous_positions = []
        queue = deque()
        queue.append(start_pos)
        visited.add(start_pos)

        while queue and len(contiguous_positions) < num_agents:
            pos = queue.popleft()
            if not valid_mask[pos]:
                continue
            contiguous_positions.append(pos)
            # Check 4 neighbors (up, down, left, right)
            neighbor_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(neighbor_deltas)
            for dr, dc in neighbor_deltas:
                nr, nc = pos[0] + dr, pos[1] + dc
                neighbor = (nr, nc)
                if (
                    0 <= nr < valid_mask.shape[0]
                    and 0 <= nc < valid_mask.shape[1]
                    and valid_mask[neighbor]
                    and neighbor not in visited
                ):
                    queue.append(neighbor)
                    visited.add(neighbor)

        if len(contiguous_positions) < num_agents:
            # Not enough contiguous positions found, return empty list
            return []
        return contiguous_positions

    def positions_in_same_area(self, level, num_agents):
        """
        Returns num_agents positions, all with valid_mask=True, all within a 5x5 square
        with the central 3x3 cut out (i.e., only the border of the 5x5 square), chosen randomly.
        Uses convolution to efficiently find valid 5x5 squares.
        If not enough such positions exist, returns [].
        """

        # Compute valid_mask as in get_valid_positions
        empty_mask = level == "empty"
        has_empty_neighbor = np.zeros_like(empty_mask, dtype=bool)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(empty_mask, shift=(dr, dc), axis=(0, 1))
            has_empty_neighbor |= shifted
        valid_mask = empty_mask & has_empty_neighbor
        valid_mask[0, :] = False
        valid_mask[-1, :] = False
        valid_mask[:, 0] = False
        valid_mask[:, -1] = False

        # Build the 5x5 border kernel (1s on border, 0s in center 3x3)
        kernel = np.zeros((5, 5), dtype=int)
        kernel[0, :] = 1
        kernel[4, :] = 1
        kernel[:, 0] = 1
        kernel[:, 4] = 1

        # Convolve valid_mask with the kernel to find all top-left corners of valid 5x5 squares
        conv = convolve2d(valid_mask.astype(int), kernel, mode="valid")
        # Find all top-left corners where all border cells are valid
        possible_squares = np.argwhere(conv >= num_agents)

        if len(possible_squares) == 0:
            return []

        # Choose a random possible square
        square_idx = random.choice(range(len(possible_squares)))
        top_left = possible_squares[square_idx]
        r0, c0 = top_left

        # List all valid positions in the 5x5 border (excluding the central 3x3)
        border_positions = []
        for dr in range(5):
            for dc in range(5):
                # Exclude central 3x3
                if 1 <= dr <= 3 and 1 <= dc <= 3:
                    continue
                rr, cc = r0 + dr, c0 + dc
                if valid_mask[rr, cc]:
                    border_positions.append((rr, cc))

        return border_positions

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
        Keep adding cylinders until *no* size/orientation fits anywhere.
        Strategy: restart the attempt with a fresh random cylinder after every
        successful placement. Stop only after ``max_consecutive_fail`` failed
        attempts *in a row* (i.e. we tried many random sizes/orientations
        without success), which strongly suggests the map is packed.
        """
        grid = np.full((self._height, self._width), "empty", dtype=object)
        self._occ[:, :] = False

        max_consecutive_fail = 2
        fails = 0
        while fails < max_consecutive_fail:
            placed = self._place_cylinder_once(grid, clearance=1)
            if placed:
                fails = 0  # reset – we still found room
            else:
                fails += 1  # try a different size/orientation

        # Finally, spawn any requested agents on leftover empty cells
        grid = self._place_agents(grid)
        return grid

    # ------------------------------------------------------------------ #
    # Cylinder placement helpers
    # ------------------------------------------------------------------ #
    def _place_cylinder_once(self, grid: np.ndarray, clearance: int = 1) -> bool:
        pat = self._generate_cylinder_pattern()
        return self._place_region(grid, pat, clearance)

    def _generate_cylinder_pattern(self) -> np.ndarray:
        length = int(self._rng.integers(2, 30))
        gap = int(self._rng.integers(1, 4))
        vertical = bool(self._rng.integers(0, 2))
        if vertical:
            h, w = length, gap + 2
            pat = np.full((h, w), "empty", dtype=object)
            pat[:, 0] = pat[:, -1] = "wall"
            pat[h // 2, 1 + gap // 2] = "altar"
        else:
            h, w = gap + 2, length
            pat = np.full((h, w), "empty", dtype=object)
            pat[0, :] = pat[-1, :] = "wall"
            pat[1 + gap // 2, w // 2] = "altar"
        return pat

    # ------------------------------------------------------------------ #
    # Agents placement (simplified)
    # ------------------------------------------------------------------ #
    def _place_agents(self, grid):
        if self._team is None:
            agents = ["agent.agent"] * self._agents
        else:
            agents = ["agent." + self._team] * self._agents

        level = np.where(~self._occupancy, "empty", "occupied")
        num_agents = len(agents)
        if self._agent_cluster_type == "no_clustering":
            valid_positions = self.get_valid_positions(level)
        elif self._agent_cluster_type == "right_next_to_each_other":
            valid_positions = self.right_next_to_each_other_positions(level, num_agents)
        elif self._agent_cluster_type == "positions_in_same_area":
            valid_positions = self.positions_in_same_area(level, num_agents)
        else:
            raise ValueError(f"Invalid agent cluster type: {self._agent_cluster_type}")

        if len(valid_positions) < num_agents:
            valid_positions = self.get_valid_positions(level)
            # raise ValueError(f"cw Not enough valid positions found for {num_agents} agents")

        random.shuffle(valid_positions)

        agent_positions = valid_positions[:num_agents]

        for a in agents:
            pos = agent_positions.pop()
            if pos:
                grid[pos] = a
                self._occ[pos] = True
        return grid

    # ------------------------------------------------------------------ #
    # Region placement utilities
    # ------------------------------------------------------------------ #
    def _place_region(self, grid, pattern: np.ndarray, clearance: int) -> bool:
        ph, pw = pattern.shape
        for r, c in self._candidate_positions((ph + 2 * clearance, pw + 2 * clearance)):
            grid[r + clearance : r + clearance + ph, c + clearance : c + clearance + pw] = pattern
            self._occ[r + clearance : r + clearance + ph, c + clearance : c + clearance + pw] |= pattern != "empty"
            return True
        return False

    def _candidate_positions(self, shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        h, w = shape
        H, W = self._occ.shape
        if H < h or W < w:
            return []
        view_shape = (H - h + 1, W - w + 1, h, w)
        sub = np.lib.stride_tricks.as_strided(self._occ, view_shape, self._occ.strides * 2)
        sums = sub.sum(axis=(2, 3))
        coords = np.argwhere(sums == 0)
        self._rng.shuffle(coords)
        return [tuple(x) for x in coords]

    def _rand_empty(self) -> Optional[Tuple[int, int]]:
        empties = np.flatnonzero(~self._occ)
        if not len(empties):
            return None
        idx = self._rng.integers(0, len(empties))
        return tuple(np.unravel_index(empties[idx], self._occ.shape))
