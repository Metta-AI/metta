"""
Defines the ConfinedRoomCoord environment.

This environment features a single rectangular room where the central area is
empty floor space for agents. Objects (mines, generators, altars) are placed
on the 1-cell thick border surrounding this floor space, excluding the corners.
The layout is designed to encourage agent coordination in a confined area.
An optional additional pure wall border can be added around the entire room.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room


class ConfinedRoomCoord(Room):
    def __init__(
        self,
        width: int,
        height: int,
        num_mines: int = 1,
        num_generators: int = 1,
        num_altars: int = 1,
        agents: Union[int, dict, DictConfig] = 1,
        # Pure wall padding around the room (floor + object border)
        border_width: int = 0,
        seed: Optional[int] = None,
    ):
        # Initialize random number generator with optional seed
        self._rng = np.random.default_rng(seed)
        self._border_width = border_width

        # User-provided width/height are for the *floor* area.
        if not (width >= 1 and height >= 1):
            raise ValueError(f"Floor dimensions (width/height) must be at least 1x1, got {width}x{height}")

        # Store configuration parameters
        self._floor_width = width
        self._floor_height = height
        self._num_mines = num_mines
        self._num_generators = num_generators
        self._num_altars = num_altars

        if isinstance(agents, int):
            if agents < 0:
                raise ValueError("Number of agents cannot be negative.")
        elif isinstance(agents, (dict, DictConfig)):
            for agent_name, count_val in agents.items():
                if not isinstance(count_val, int) or count_val < 0:
                    raise ValueError(
                        f"Agent count for '{str(agent_name)}' must be a non-negative integer, got {count_val}"
                    )
        else:
            raise TypeError(f"Agents parameter must be an int, dict, or DictConfig, got {type(agents)}")
        self._agents_spec = agents

        # Calculate room dimensions:
        # Core room = floor + 1-cell functional border (2 cells in each dim)
        core_width = self._floor_width + 2
        core_height = self._floor_height + 2

        # Total grid dimensions = core + optional pure wall border
        actual_grid_width = core_width + 2 * self._border_width
        actual_grid_height = core_height + 2 * self._border_width

        # Initialize parent Room class with wall border and room label
        super().__init__(border_width=self._border_width, border_object="wall", labels=["confined_room_coord"])

        # Set final grid dimensions
        self._width = actual_grid_width
        self._height = actual_grid_height

        # Initialize occupancy grid (tracks which cells are occupied)
        self._occ = np.zeros((self._height, self._width), dtype=bool)
        self.set_size_labels(self._width, self._height)

    def _build(self) -> np.ndarray:
        # Initialize empty grid and reset occupancy tracking
        grid = np.full((self._height, self._width), "empty", dtype=object)
        self._occ.fill(False)

        # 1. Create outer pure wall border if specified
        if self._border_width > 0:
            for i in range(self._border_width):
                # Place walls on all four sides
                grid[i, :] = "wall"
                self._occ[i, :] = True  # Top border
                grid[self._height - 1 - i, :] = "wall"
                self._occ[self._height - 1 - i, :] = True  # Bottom border
                grid[:, i] = "wall"
                self._occ[:, i] = True  # Left border
                grid[:, self._width - 1 - i] = "wall"
                self._occ[:, self._width - 1 - i] = True  # Right border

        # 2. Calculate coordinates for the functional border and its corners
        # The functional border is the 1-cell thick border around the floor area
        fb_top_row = self._border_width
        fb_bottom_row = self._height - 1 - self._border_width
        fb_left_col = self._border_width
        fb_right_col = self._width - 1 - self._border_width

        functional_border_cells: List[Tuple[int, int]] = []

        # Identify corner cells of the functional border
        # These will always be walls and are not available for objects.
        fb_corners = []
        if fb_top_row <= fb_bottom_row and fb_left_col <= fb_right_col:  # Check if functional area has any size
            # Add all four corners of the functional border
            fb_corners.extend(
                [
                    (fb_top_row, fb_left_col),
                    (fb_top_row, fb_right_col),
                    (fb_bottom_row, fb_left_col),
                    (fb_bottom_row, fb_right_col),
                ]
            )
            fb_corners = sorted(list(set(fb_corners)))  # Unique corners

            # Collect non-corner functional border cells
            # Top row (excluding corners)
            for c in range(fb_left_col + 1, fb_right_col):
                functional_border_cells.append((fb_top_row, c))
            # Bottom row (excluding corners)
            if fb_bottom_row > fb_top_row:  # Avoid double-counting for 1-row high functional border
                for c in range(fb_left_col + 1, fb_right_col):
                    functional_border_cells.append((fb_bottom_row, c))
            # Left column (excluding corners)
            for r in range(fb_top_row + 1, fb_bottom_row):
                functional_border_cells.append((r, fb_left_col))
            # Right column (excluding corners)
            if fb_right_col > fb_left_col:  # Avoid double-counting for 1-col wide functional border
                for r in range(fb_top_row + 1, fb_bottom_row):
                    functional_border_cells.append((r, fb_right_col))

        functional_border_cells = sorted(list(set(functional_border_cells)))  # Unique non-corner cells
        # Shuffle for random object placement
        self._rng.shuffle(functional_border_cells)

        # 2a. Check if enough space for all objects on non-corner border
        total_objects_to_place = self._num_mines + self._num_generators + self._num_altars
        available_non_corner_slots = len(functional_border_cells)
        if total_objects_to_place > available_non_corner_slots:
            print(
                f"Warning: Requested {total_objects_to_place} total objects "
                f"but only {available_non_corner_slots} non-corner border slots are available. "
                f"Not all objects may be placed."
            )

        # 3. Place game objects (mines, generators, altars) on the non-corner functional border
        objects_to_place = (
            [("mine_red", self._num_mines)] + [("generator_red", self._num_generators)] + [("altar", self._num_altars)]
        )

        temp_available_non_corner_fb_cells = list(functional_border_cells)  # Copy for modification

        cells_used_for_objects = []

        for obj_name, count in objects_to_place:
            placed_count = 0
            for _ in range(count):
                if not temp_available_non_corner_fb_cells:
                    break
                pos = temp_available_non_corner_fb_cells.pop(0)
                grid[pos] = obj_name
                self._occ[pos] = True
                placed_count += 1
                cells_used_for_objects.append(pos)
            if placed_count < count:
                print(f"Warning: Could only place {placed_count}/{count} of '{obj_name}' on non-corner border.")

        # 4. Fill functional border corners and remaining non-corner cells with "wall"
        # Place walls in the identified corners
        for r_c, c_c in fb_corners:
            if 0 <= r_c < self._height and 0 <= c_c < self._width:  # Boundary check
                if grid[r_c, c_c] == "empty":  # Ensure not overwriting pure border if border_width is 0
                    grid[r_c, c_c] = "wall"
                    self._occ[r_c, c_c] = True

        # Fill remaining non-corner functional border cells (those not used for objects) with "wall"
        for pos in functional_border_cells:  # Iterate over original shuffled list
            if pos not in cells_used_for_objects:  # If this slot wasn't taken by an object
                if 0 <= pos[0] < self._height and 0 <= pos[1] < self._width:  # Boundary check
                    if grid[pos] == "empty":  # Ensure not overwriting pure border if border_width is 0
                        grid[pos] = "wall"
                        self._occ[pos] = True

        # 5. Identify the central floor area for agent placement
        # Floor is inside the functional border.
        floor_tl_r = self._border_width + 1
        floor_tl_c = self._border_width + 1
        # The functional border is 1-thick, so floor ends 1 cell before its outer edge.
        floor_br_r = (self._height - 1 - self._border_width) - 1
        floor_br_c = (self._width - 1 - self._border_width) - 1

        empty_floor_cells: List[Tuple[int, int]] = []
        if floor_tl_r <= floor_br_r and floor_tl_c <= floor_br_c:
            for r_f in range(floor_tl_r, floor_br_r + 1):
                for c_f in range(floor_tl_c, floor_br_c + 1):
                    # Cells in this area should be "empty" unless something went wrong
                    if grid[r_f, c_f] == "empty":
                        empty_floor_cells.append((r_f, c_f))

        self._rng.shuffle(empty_floor_cells)

        # 6. Place agents in the floor area
        agent_symbols_to_place: List[str] = []
        if isinstance(self._agents_spec, int):
            count = self._agents_spec
            # Agent type defaults to "agent.agent" if an integer is provided
            agent_symbols_to_place = ["agent.agent"] * count
        elif isinstance(self._agents_spec, (dict, DictConfig)):
            for name, count_val in self._agents_spec.items():
                # Ensure name is treated as a string for processing
                s_name = str(name)
                processed_name = s_name if "." in s_name else f"agent.{s_name}"
                agent_symbols_to_place.extend([processed_name] * count_val)
            self._rng.shuffle(agent_symbols_to_place)

        agents_placed_count = 0
        for i in range(min(len(agent_symbols_to_place), len(empty_floor_cells))):
            pos = empty_floor_cells[i]
            grid[pos] = agent_symbols_to_place[i]
            self._occ[pos] = True
            agents_placed_count += 1

        if agents_placed_count < len(agent_symbols_to_place):
            print(
                f"Warning: Not enough empty floor cells. "
                f"Placed {agents_placed_count}/{len(agent_symbols_to_place)} agents."
            )

        return grid
