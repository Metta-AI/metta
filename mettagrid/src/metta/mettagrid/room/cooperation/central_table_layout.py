"""
Defines the CentralTableLayout room environment.

This environment features a central rectangular "table" made of walls,
surrounded by a lane where agents can move and interact with objects.
Objects like mines, generators, and altars are placed within this lane.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room


class CentralTableLayout(Room):
    STYLE_PARAMETERS = {
        "default": {
            "lane_width": 1,
            "num_mines": 4,
            "num_generators": 2,
            "num_altars": 2,
        },
    }

    def __init__(
        self,
        width: int,  # User-provided width for the core functional area
        height: int,  # User-provided height for the core functional area
        agents: Union[int, dict, DictConfig] = 1,
        seed: Optional[int] = None,
        lane_width: int = 1,
        num_mines: int = 2,
        num_generators: int = 2,
        num_altars: int = 2,
        style: str = "default",
        border_width: int = 0,
    ):
        self._rng = np.random.default_rng(seed)
        self._border_width = border_width

        # User-provided width/height are for the core area.
        # Style might override lane_width, so fetch it first to use in validation.
        effective_lane_width_for_validation = lane_width
        if style != "default" and style in self.STYLE_PARAMETERS:
            params = self.STYLE_PARAMETERS[style]
            effective_lane_width_for_validation = params.get("lane_width", lane_width)

        # Minimum dimension for the core structure:
        # It needs space for:
        #   - 1-cell thick functional border on each side (total 2 cells)
        #   - lane_width on each side (total 2 * lane_width cells)
        #   - A minimum 1x1 central table (total 1 cell)
        min_core_dimension = 2 * (effective_lane_width_for_validation + 1) + 1

        if width < min_core_dimension or height < min_core_dimension:
            raise ValueError(
                f"Core dimensions ({width}x{height}) are too small for the specified "
                f"lane_width ({effective_lane_width_for_validation}). "
                f"Minimum core dimension required is {min_core_dimension}x{min_core_dimension} "
                f"(this is for the functional border, lane, and a 1x1 central table, "
                f"excluding the pure wall border_width)."
            )

        # Actual grid dimensions for internal use and for `super().__init__`
        # These include the extra padding from border_width.
        actual_grid_width = width + 2 * self._border_width
        actual_grid_height = height + 2 * self._border_width

        # CORRECTED CALL TO SUPER: Removed width and height arguments
        super().__init__(border_width=self._border_width, border_object="wall", labels=["central_table", style])

        # Store the *actual total* dimensions for the grid generation
        self._width = actual_grid_width
        self._height = actual_grid_height

        # Store agents and team info
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

        # Initialize and apply style parameters for items and the final self._lane_width
        self._lane_width = lane_width  # Start with user-passed, style might override
        self._num_mines = num_mines
        self._num_generators = num_generators
        self._num_altars = num_altars

        if style != "default" and style in self.STYLE_PARAMETERS:
            params = self.STYLE_PARAMETERS[style]
            self._lane_width = params.get("lane_width", lane_width)  # Final lane_width
            self._num_mines = params.get("num_mines", num_mines)
            self._num_generators = params.get("num_generators", num_generators)
            self._num_altars = params.get("num_altars", num_altars)

        # Initialize occupancy grid with actual total dimensions
        self._occ = np.zeros((self._height, self._width), dtype=bool)
        # Set size labels based on actual total dimensions
        self.set_size_labels(self._width, self._height)

    def _build(self) -> np.ndarray:
        grid = np.full((self._height, self._width), "empty", dtype=object)
        self._occ.fill(False)

        # 0. Draw outer pure wall border if border_width > 0
        if self._border_width > 0:
            for i in range(self._border_width):
                # Top border layer
                grid[i, :] = "wall"
                self._occ[i, :] = True
                # Bottom border layer
                grid[self._height - 1 - i, :] = "wall"
                self._occ[self._height - 1 - i, :] = True
                # Left border layer
                grid[:, i] = "wall"
                self._occ[:, i] = True
                # Right border layer
                grid[:, self._width - 1 - i] = "wall"
                self._occ[:, self._width - 1 - i] = True

        # 1. Define and draw the central table (walls)
        # Coordinates are now relative to the area inside the pure wall border
        # Effective top-left of the 'functional' area (where objects/lanes/table go)
        func_area_tl_r = self._border_width
        func_area_tl_c = self._border_width

        # Table top-left corner, offset by lane_width and 1-cell for functional border
        table_tl_r = func_area_tl_r + self._lane_width + 1
        table_tl_c = func_area_tl_c + self._lane_width + 1

        # Table bottom-right corner
        # Effective bottom-right of the 'functional' area
        func_area_br_r = self._height - 1 - self._border_width
        func_area_br_c = self._width - 1 - self._border_width

        table_br_r = func_area_br_r - self._lane_width - 1
        table_br_c = func_area_br_c - self._lane_width - 1

        if table_tl_r > table_br_r or table_tl_c > table_br_c:
            raise ValueError(
                f"Calculated table dimensions are invalid or non-existent. "
                f"Grid: {self._width}x{self._height}, border: {self._border_width}, lane: {self._lane_width}. "
                f"Table TL:({table_tl_r},{table_tl_c}), BR:({table_br_r},{table_br_c}). "
                f"Functional Area TL:({func_area_tl_r},{func_area_tl_c}), BR:({func_area_br_r},{func_area_br_c})"
            )
        grid[table_tl_r : table_br_r + 1, table_tl_c : table_br_c + 1] = "wall"
        self._occ[table_tl_r : table_br_r + 1, table_tl_c : table_br_c + 1] = True

        # 2. Identify Object Placement Slots (on the 'functional' outer border)
        # This is the layer immediately inside the pure wall border (if any)
        # and outside the lane.

        # Coordinates for the functional border
        fb_top_row = func_area_tl_r
        fb_bottom_row = func_area_br_r
        fb_left_col = func_area_tl_c
        fb_right_col = func_area_br_c

        border_cells_map = {
            "top": [],
            "bottom": [],
            "left": [],
            "right": [],
            "top_left_corner": (fb_top_row, fb_left_col),
            "top_right_corner": (fb_top_row, fb_right_col),
            "bottom_left_corner": (fb_bottom_row, fb_left_col),
            "bottom_right_corner": (fb_bottom_row, fb_right_col),
        }

        # Populate functional border segments (excluding corners initially)
        # Top functional border: (fb_top_row, c_idx) for c_idx from fb_left_col + 1 to fb_right_col - 1
        for c_idx in range(fb_left_col + 1, fb_right_col):
            border_cells_map["top"].append((fb_top_row, c_idx))
        # Bottom functional border
        if fb_bottom_row > fb_top_row:  # ensure there is space for distinct top and bottom
            for c_idx in range(fb_left_col + 1, fb_right_col):
                border_cells_map["bottom"].append((fb_bottom_row, c_idx))
        # Left functional border
        for r_idx in range(fb_top_row + 1, fb_bottom_row):
            border_cells_map["left"].append((r_idx, fb_left_col))
        # Right functional border
        if fb_right_col > fb_left_col:  # ensure there is space for distinct left and right
            for r_idx in range(fb_top_row + 1, fb_bottom_row):
                border_cells_map["right"].append((r_idx, fb_right_col))

        outer_corners_list = sorted(
            list(
                set(
                    [
                        border_cells_map["top_left_corner"],
                        border_cells_map["top_right_corner"],
                        border_cells_map["bottom_left_corner"],
                        border_cells_map["bottom_right_corner"],
                    ]
                )
            )
        )

        # Ensure corners are valid (not overlapping if dimensions are too small)
        # This might occur if func_area is very small, e.g., 1xN or Nx1.
        # For a 1x1 functional area, all corners are the same point.
        # The set conversion above handles this implicitly by storing only unique points.

        all_potential_border_slots = []
        for seg_key in ["top", "bottom", "left", "right"]:
            all_potential_border_slots.extend(border_cells_map[seg_key])
        all_potential_border_slots.extend(outer_corners_list)
        all_potential_border_slots = sorted(list(set(all_potential_border_slots)))

        # Determine preferred border segments based on table dimensions
        table_actual_width = table_br_c - table_tl_c + 1
        table_actual_height = table_br_r - table_tl_r + 1

        if table_actual_width >= table_actual_height:  # Table is wider or square
            generator_preferred_border_segments = border_cells_map["top"] + border_cells_map["bottom"]
            mine_altar_preferred_border_segments = border_cells_map["left"] + border_cells_map["right"]
        else:  # Table is taller
            generator_preferred_border_segments = border_cells_map["left"] + border_cells_map["right"]
            mine_altar_preferred_border_segments = border_cells_map["top"] + border_cells_map["bottom"]

        # Helper to place items on the functional border
        def _place_item_on_border(item_name: str, count: int, preferred_segments: List[Tuple[int, int]]):
            placed_count = 0
            candidate_slots = []
            current_preferred_slots = list(set(preferred_segments))
            self._rng.shuffle(current_preferred_slots)
            candidate_slots.extend(current_preferred_slots)
            current_corner_slots = [c for c in outer_corners_list if c not in candidate_slots]
            self._rng.shuffle(current_corner_slots)
            candidate_slots.extend(current_corner_slots)
            other_border_slots = [s for s in all_potential_border_slots if s not in candidate_slots]
            self._rng.shuffle(other_border_slots)
            candidate_slots.extend(other_border_slots)

            for pos in candidate_slots:
                if placed_count == count:
                    break
                if 0 <= pos[0] < self._height and 0 <= pos[1] < self._width and grid[pos] == "empty":
                    grid[pos] = item_name
                    self._occ[pos] = True
                    placed_count += 1
            if placed_count < count:
                print(f"Warning: Could only place {placed_count}/{count} of '{item_name}' on functional border.")

        _place_item_on_border("generator_red", self._num_generators, generator_preferred_border_segments)
        _place_item_on_border("mine_red", self._num_mines, mine_altar_preferred_border_segments)
        _place_item_on_border("altar", self._num_altars, mine_altar_preferred_border_segments)

        # 4. Fill remaining "empty" functional border cells with "wall"
        for pos in all_potential_border_slots:
            if 0 <= pos[0] < self._height and 0 <= pos[1] < self._width and grid[pos] == "empty":
                grid[pos] = "wall"
                self._occ[pos] = True

        # 5. Identify Lane Cells for agent placement
        # Lane cells are between the functional border and the central table.
        lane_cells_list = []
        # Lane starts 1 cell inside the functional border
        lane_tl_r = func_area_tl_r + 1
        lane_tl_c = func_area_tl_c + 1
        lane_br_r = func_area_br_r - 1
        lane_br_c = func_area_br_c - 1

        for r_lane in range(lane_tl_r, lane_br_r + 1):
            for c_lane in range(lane_tl_c, lane_br_c + 1):
                # A cell is a lane cell if it's "empty" AND not part of the table
                # (The table is already on the grid, so checking grid[pos] == "empty" suffices)
                if grid[r_lane, c_lane] == "empty":
                    lane_cells_list.append((r_lane, c_lane))

        self._rng.shuffle(lane_cells_list)

        # 6. Place agents in these empty lane_cells_list
        agent_symbols_to_place = []
        if isinstance(self._agents_spec, int):
            count = self._agents_spec
            # Agent type defaults to "agent.agent" if an integer is provided
            agent_symbols_to_place = ["agent.agent"] * count
        elif isinstance(self._agents_spec, (dict, DictConfig)):
            for name, agent_count in self._agents_spec.items():
                # Ensure name is treated as a string for processing
                s_name = str(name)
                processed_name = s_name if "." in s_name else f"agent.{s_name}"
                agent_symbols_to_place.extend([processed_name] * agent_count)
            self._rng.shuffle(agent_symbols_to_place)

        for i in range(min(len(agent_symbols_to_place), len(lane_cells_list))):
            pos = lane_cells_list[i]
            grid[pos] = agent_symbols_to_place[i]
            self._occ[pos] = True

        if len(agent_symbols_to_place) > len(lane_cells_list):
            print(
                f"Warning: Not enough empty lane cells to place all agents. "
                f"Placed {len(lane_cells_list)}/{len(agent_symbols_to_place)}."
            )
        return grid

    # Example of how to use it later:
    # room = CentralTableLayout(width=11, height=8, agents=2, lane_width=1,
    #                           num_mines=4, num_generators=2, num_altars=2, seed=42)
    # grid_array = room.build() # or room.grid if auto-build in Room base
