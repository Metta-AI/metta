from typing import Set, Tuple
import numpy as np
from omegaconf import DictConfig

# Import the base Room class from mettagrid.
from mettagrid.config.room.room import Room

class MultiChamberRoom(Room):
    """
    A room divided horizontally into four chambers by three full-width barriers with randomized gaps.
    
    Chamber layout (from top to bottom):
      - Top Chamber: Contains the agent.
      - Second Chamber: Contains the generator.
      - Third Chamber: Contains the converter.
      - Bottom Chamber: Contains the heart altar.
    
    For each barrier (drawn at the rows specified in barrier_rows), a gap of specified width is
    placed at a random column (within the allowed range).
    """
    def __init__(
        self,
        width: int,
        height: int,
        multi_chamber_params: DictConfig,
        agents: int = 1,
        seed=None,
        border_width: int = 1,
        border_object: str = "wall",
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._width = width
        self._height = height
        self._params = multi_chamber_params
        self._agents = agents
        self._rng = np.random.default_rng(seed)
        self._grid = np.full((self._height, self._width), "empty", dtype='<U50')
        self._placed_positions: Set[Tuple[int, int]] = set()

    def _build(self) -> np.ndarray:
        # 1. Place the outer border walls.
        self._place_border()

        # 2. Retrieve barrier configuration.
        barrier_rows = self._params.get("barrier_rows", [])
        gap_width = self._params.get("gap_width", 1)
        bw = self._border_width

        # 3. For each barrier, choose a random gap column in the valid region.
        for barrier_row in barrier_rows:
            gap_col = int(self._rng.integers(bw, self._width - bw - gap_width + 1))
            self._draw_barrier_with_gap(barrier_row, gap_col, gap_width)

        # 4. Determine chamber ranges based on the barriers.
        # Chambers:
        #   - Top Chamber: from just below the top border to barrier_rows[0]-1.
        #   - Intermediate Chambers: barrier_rows[i]+1 to barrier_rows[i+1]-1.
        #   - Bottom Chamber: barrier_rows[-1]+1 to just above the bottom border.
        chamber_ranges = []
        if barrier_rows:
            chamber_ranges.append((bw, barrier_rows[0] - 1))  # Top chamber
            for i in range(len(barrier_rows) - 1):
                chamber_ranges.append((barrier_rows[i] + 1, barrier_rows[i+1] - 1))
            chamber_ranges.append((barrier_rows[-1] + 1, self._height - bw - 1))  # Bottom chamber
        else:
            chamber_ranges.append((bw, self._height - bw - 1))

        # 5. Retrieve object labels for each chamber.
        # Expected keys: "top", "second", "third", "bottom"
        chambers = self._params.get("chambers", {})
        ordered_keys = ["top", "second", "third", "bottom"]
        objects = []
        for key in ordered_keys:
            if key in chambers:
                objects.append(chambers[key])
        if len(objects) != len(chamber_ranges):
            raise ValueError("Number of chamber objects does not match the number of chambers defined by barriers.")

        # 6. Place each object in the center of its respective chamber.
        for (row_start, row_end), obj in zip(chamber_ranges, objects):
            center_row = (row_start + row_end) // 2
            center_col = self._width // 2
            self._grid[center_row, center_col] = obj
            self._placed_positions.add((center_col, center_row))

        return self._grid

    def _place_border(self) -> None:
        bw = self._border_width
        # Top and bottom borders.
        for x in range(self._width):
            for y in range(bw):
                self._grid[y, x] = self._border_object
                self._grid[self._height - 1 - y, x] = self._border_object
        # Left and right borders.
        for y in range(self._height):
            for x in range(bw):
                self._grid[y, x] = self._border_object
                self._grid[y, self._width - 1 - x] = self._border_object

    def _draw_barrier_with_gap(self, barrier_row: int, gap_col: int, gap_width: int) -> None:
        """
        Draw a horizontal barrier at the specified row.
        The barrier spans from the left to right border except for the gap region.
        """
        bw = self._border_width
        x_start = bw
        x_end = self._width - bw - 1  # inclusive
        for x in range(x_start, x_end + 1):
            if gap_col <= x < gap_col + gap_width:
                self._grid[barrier_row, x] = "empty"
            else:
                self._grid[barrier_row, x] = "wall"
                self._placed_positions.add((x, barrier_row))
