from __future__ import annotations

from typing import Tuple

import numpy as np

from metta.common.util.config import Config
from metta.map.scene import Scene


class CoopDividedParams(Config):
    """
    Parameters for the CoopDivided scene.

    - agents_top: Number of agents to place in the top half of the room
    - agents_bottom: Number of agents to place in the bottom half of the room
    - generator_x: Optional fixed x (column) for the generator gap. If None, choose randomly.
    """

    agents_top: int = 1
    agents_bottom: int = 1
    generator_x: int | None = None


class CoopDivided(Scene[CoopDividedParams]):
    """
    Divide the room into two halves with a horizontal wall and place a generator in
    the wall as the only interaction point. The generator's x-position is randomized
    (unless provided) along the dividing wall. Agents and their objectives are
    separated: mine + agent(s) on top, altar + agent(s) on bottom.
    """

    def post_init(self) -> None:  # noqa: D401
        # Ensure params are validated/instantiated
        self.params = self.validate_params(self.params)

    def _random_empty_in_rows(self, row_min: int, row_max: int) -> Tuple[int, int]:
        mask = self.grid == "empty"
        # Restrict to row range [row_min, row_max]
        row_indices = np.arange(self.height)
        valid_rows = (row_indices >= row_min) & (row_indices <= row_max)
        mask &= valid_rows[:, None]

        coords = np.argwhere(mask)
        if coords.shape[0] == 0:
            # Fallback: pick any empty cell anywhere to avoid crashes
            coords = np.argwhere(self.grid == "empty")
        assert coords.shape[0] > 0, "No empty cells available to place object"
        r, c = coords[self.rng.integers(0, coords.shape[0])]
        return int(r), int(c)

    def render(self) -> None:
        # Start from the provided grid (should be 'empty' inside borders)
        # Draw the horizontal dividing wall with a single generator gap
        mid_row = self.height // 2

        # Choose generator x position (avoid outer borders)
        if self.params.generator_x is None:
            gen_x = int(self.rng.integers(1, max(1, self.width - 1)))
            # Ensure not placing on outer borders
            gen_x = min(max(gen_x, 1), self.width - 2)
        else:
            gen_x = int(self.params.generator_x)
            gen_x = min(max(gen_x, 1), self.width - 2)

        # Fill the mid row with walls, then carve generator cell
        for x in range(self.width):
            self.grid[mid_row, x] = "wall"
        self.grid[mid_row, gen_x] = "generator_red"

        # Ensure adjacency above/below generator is open for interaction
        if mid_row - 1 >= 0:
            self.grid[mid_row - 1, gen_x] = "empty"
        if mid_row + 1 < self.height:
            self.grid[mid_row + 1, gen_x] = "empty"

        # Place mine in the top half
        if mid_row - 1 >= 0:
            r_mine, c_mine = self._random_empty_in_rows(0, mid_row - 1)
            self.grid[r_mine, c_mine] = "mine_red"

        # Place altar in the bottom half
        if mid_row + 1 < self.height:
            r_altar, c_altar = self._random_empty_in_rows(mid_row + 1, self.height - 1)
            self.grid[r_altar, c_altar] = "altar"

        # Place agents: top half and bottom half
        for _ in range(max(0, self.params.agents_top)):
            if mid_row - 1 >= 0:
                r, c = self._random_empty_in_rows(0, mid_row - 1)
                self.grid[r, c] = "agent.team_1"

        for _ in range(max(0, self.params.agents_bottom)):
            if mid_row + 1 < self.height:
                r, c = self._random_empty_in_rows(mid_row + 1, self.height - 1)
                self.grid[r, c] = "agent.team_1"
