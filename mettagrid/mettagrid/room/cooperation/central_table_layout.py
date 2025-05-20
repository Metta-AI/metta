"""
Defines the CentralTableLayout room environment.

This environment features a central rectangular "table" made of walls,
surrounded by a lane where agents can move and interact with objects.
Objects like mines, generators, and altars are placed within this lane.
"""

from typing import List, Optional, Tuple, Dict, Union
import numpy as np
from mettagrid.room.room import Room


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
        width: int,
        height: int,
        agents: Union[int, Dict[str, int]] = 1,
        seed: Optional[int] = None,
        lane_width: int = 1,
        num_mines: int = 2,
        num_generators: int = 2,
        num_altars: int = 2,
        # For simple team assignment if agents is int
        team: Optional[str] = None,
        style: str = "default",
    ):
        super().__init__(border_width=0, border_object="wall",
                         labels=["central_table", style])
        self._rng = np.random.default_rng(seed)

        # Check if the grid dimensions are too small for the specified lane_width
        if width < 2 * (lane_width + 1) + 1 or height < 2 * (lane_width + 1) + 1:
            raise ValueError(
                f"Grid dimensions ({width}x{height}) are too small for the specified lane_width ({lane_width}) "
                f"and a minimum 1x1 table. Minimum {2 * (lane_width + 1) + 1}x{2 * (lane_width + 1) + 1} required."
            )

        # Initialize room dimensions
        self._width = width
        self._height = height
        # Initialize agents specification
        self._agents_spec = agents
        self._team = team
        # Initialize style parameters
        self._lane_width = lane_width
        self._num_mines = num_mines
        self._num_generators = num_generators
        self._num_altars = num_altars

        # Apply style parameters if specified style is valid and not "default" (which uses args)
        if style != "default" and style in self.STYLE_PARAMETERS:
            params = self.STYLE_PARAMETERS[style]
            self._lane_width = params.get("lane_width", lane_width)
            self._num_mines = params.get("num_mines", num_mines)
            self._num_generators = params.get("num_generators", num_generators)
            self._num_altars = params.get("num_altars", num_altars)

        # Initialize occupancy grid
        self._occ = np.zeros((height, width), dtype=bool)
        # Set size labels
        self.set_size_labels(width, height)

    def _build(self) -> np.ndarray:
        # Create a grid filled with "empty" objects
        grid = np.full((self._height, self._width), "empty", dtype=object)
        # Fill the occupancy grid with False values
        self._occ.fill(False)

        # 1. Define and draw the central table (these are walls)
        # Top-left corner (inclusive) of the table.
        table_tl_r = self._lane_width + 1
        table_tl_c = self._lane_width + 1
        # Bottom-right corner (inclusive) of the table.
        # The -2 accounts for the 1-cell outer border and 1-cell lane.
        table_br_r = self._height - self._lane_width - 2
        table_br_c = self._width - self._lane_width - 2

        if table_tl_r > table_br_r or table_tl_c > table_br_c:
            raise ValueError(
                f"Calculated table dimensions are invalid or non-existent. "
                f"Grid: {self._width}x{self._height}, lane: {self._lane_width}. "
                f"Table TL:({table_tl_r},{table_tl_c}), BR:({table_br_r},{table_br_c})"
            )

        grid[table_tl_r: table_br_r + 1, table_tl_c: table_br_c + 1] = "wall"
        self._occ[table_tl_r: table_br_r + 1,
                  table_tl_c: table_br_c + 1] = True

        # 2. Identify Object Placement Slots (on the absolute outer border)
        border_cells_map = {
            "top": [], "bottom": [], "left": [], "right": [],
            "top_left_corner": (0, 0),
            "top_right_corner": (0, self._width-1) if self._width > 0 else (0, 0),
            "bottom_left_corner": (self._height-1, 0) if self._height > 0 else (0, 0),
            "bottom_right_corner": (self._height-1, self._width-1) if self._height > 0 and self._width > 0 else (0, 0)
        }

        # Populate border segments, excluding corners initially
        for c_idx in range(1, self._width - 1):  # Top border (0, c_idx)
            border_cells_map["top"].append((0, c_idx))
        if self._height > 1:  # Bottom border (self._height-1, c_idx)
            for c_idx in range(1, self._width - 1):
                border_cells_map["bottom"].append((self._height - 1, c_idx))
        for r_idx in range(1, self._height - 1):  # Left border (r_idx, 0)
            border_cells_map["left"].append((r_idx, 0))
        if self._width > 1:  # Right border (r_idx, self._width-1)
            for r_idx in range(1, self._height - 1):
                border_cells_map["right"].append((r_idx, self._width - 1))

        outer_corners_list = sorted(list(set([  # Get unique corner coordinates
            border_cells_map["top_left_corner"], border_cells_map["top_right_corner"],
            border_cells_map["bottom_left_corner"], border_cells_map["bottom_right_corner"]
        ])))

        all_potential_border_slots = []
        for seg_key in ["top", "bottom", "left", "right"]:
            all_potential_border_slots.extend(border_cells_map[seg_key])
        all_potential_border_slots.extend(outer_corners_list)
        # Ensure uniqueness and sort, mainly for deterministic behavior if rng is not used or for easier debugging
        all_potential_border_slots = sorted(
            list(set(all_potential_border_slots)))

        # Determine preferred border segments for items based on table dimensions
        # This refers to the dimensions of the central table itself.
        table_actual_width = table_br_c - table_tl_c + 1
        table_actual_height = table_br_r - table_tl_r + 1

        if table_actual_width >= table_actual_height:  # Table is wider or square
            # Generators on top/bottom border segments
            generator_preferred_border_segments = border_cells_map["top"] + \
                border_cells_map["bottom"]
            # Mines/Altars on left/right border segments
            mine_altar_preferred_border_segments = border_cells_map["left"] + \
                border_cells_map["right"]
        else:  # Table is taller
            # Generators on left/right border segments
            generator_preferred_border_segments = border_cells_map["left"] + \
                border_cells_map["right"]
            # Mines/Altars on top/bottom border segments
            mine_altar_preferred_border_segments = border_cells_map["top"] + \
                border_cells_map["bottom"]

        # Helper to place items on the border
        def _place_item_on_border(item_name: str, count: int, preferred_segments: List[Tuple[int, int]]):
            placed_count = 0

            # Build a candidate list: preferred (shuffled), then corners (shuffled), then all others (shuffled)
            candidate_slots = []

            # Add shuffled preferred segments
            current_preferred_slots = list(set(preferred_segments))  # Unique
            self._rng.shuffle(current_preferred_slots)
            candidate_slots.extend(current_preferred_slots)

            # Add shuffled corners not already in preferred
            current_corner_slots = [
                c for c in outer_corners_list if c not in candidate_slots]
            self._rng.shuffle(current_corner_slots)
            candidate_slots.extend(current_corner_slots)

            # Add all other border slots not yet included, also shuffled
            # This ensures all border slots are considered if preferred/corners are not enough or too many.
            other_border_slots = [
                s for s in all_potential_border_slots if s not in candidate_slots]
            self._rng.shuffle(other_border_slots)
            candidate_slots.extend(other_border_slots)

            for pos in candidate_slots:
                if placed_count == count:
                    break
                # Check if the border cell is available (still "empty")
                if grid[pos] == "empty":
                    grid[pos] = item_name
                    self._occ[pos] = True
                    placed_count += 1

            if placed_count < count:
                print(
                    f"Warning: Could only place {placed_count}/{count} of '{item_name}' on border.")

        _place_item_on_border("generator", self._num_generators,
                              generator_preferred_border_segments)
        _place_item_on_border("mine", self._num_mines,
                              mine_altar_preferred_border_segments)
        _place_item_on_border("altar", self._num_altars,
                              mine_altar_preferred_border_segments)

        # 4. Fill remaining "empty" border cells with "wall"
        for pos in all_potential_border_slots:  # Iterate over all defined border positions
            # If not used for an item (mine, generator, altar)
            if grid[pos] == "empty":
                grid[pos] = "wall"
                self._occ[pos] = True  # Mark as occupied by a wall

        # 5. Identify Lane Cells for agent placement
        # These are cells inside the outer border but not part of the central table.
        # They should currently be "empty".
        lane_cells_list = []
        # Iterate from 1 to height-2 / width-2 to exclude the outer border cells.
        for r_lane in range(1, self._height - 1):
            for c_lane in range(1, self._width - 1):
                # A cell is a lane cell if it's not part of the central table
                # (which is already marked on the grid and _occ) and it's inside the outer border.
                # Since the table is already placed, and border objects/walls are placed,
                # any remaining "empty" cell in this inner region is a lane cell.
                if grid[r_lane, c_lane] == "empty":
                    lane_cells_list.append((r_lane, c_lane))

        # Shuffle for random agent placement
        self._rng.shuffle(lane_cells_list)

        # 6. Place agents in these empty lane_cells_list
        agent_symbols_to_place = []
        if isinstance(self._agents_spec, int):
            count = self._agents_spec
            if self._team:
                agent_symbols_to_place = [f"agent.{self._team}"] * count
            else:
                agent_symbols_to_place = ["agent.agent"] * count
        elif isinstance(self._agents_spec, dict):
            for name, agent_count in self._agents_spec.items():
                processed_name = name if "." in name else f"agent.{name}"
                agent_symbols_to_place.extend([processed_name] * agent_count)
            # Shuffle if multiple agent types defined in dict
            self._rng.shuffle(agent_symbols_to_place)

        for i in range(min(len(agent_symbols_to_place), len(lane_cells_list))):
            pos = lane_cells_list[i]
            grid[pos] = agent_symbols_to_place[i]
            self._occ[pos] = True  # Mark agent position as occupied

        if len(agent_symbols_to_place) > len(lane_cells_list):
            print(f"Warning: Not enough empty lane cells to place all agents. "
                  f"Placed {len(lane_cells_list)}/{len(agent_symbols_to_place)} agents.")

        return grid

    # Example of how to use it later:
    # room = CentralTableLayout(width=11, height=8, agents=2, lane_width=1,
    #                           num_mines=4, num_generators=2, num_altars=2, seed=42)
    # grid_array = room.build() # or room.grid if auto-build in Room base
