"""
Defines the TwoRoomsCoord room environment.

This environment features two distinct rectangular rooms of the same floor dimensions,
separated by a single shared wall. This shared wall can contain "generator" objects.
One room is randomly designated to have "altar" objects, and the other
room will have "mine" objects.
Agents are placed alternately in the available floor spaces of the two rooms.
The objective is to facilitate tasks requiring cooperation through the shared
generators (e.g., mine -> generator -> altar sequence).
"""

from typing import List, Optional, Set, Tuple, Union

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room


class TwoRoomsCoord(Room):
    def __init__(
        self,
        # (width, height) of empty floor space for EACH room
        width: int,
        height: int,
        num_shared_generators: int,
        num_altars: int,
        num_mines: int,
        agents: Union[int, dict, DictConfig] = 2,
        # "horizontal", "vertical", or None for random
        arrangement: Optional[str] = None,
        border_width: int = 0,
        seed: Optional[int] = None,
    ):
        self._rng = np.random.default_rng(seed)
        self._border_width = border_width

        if not (width >= 3 and height >= 3):
            raise ValueError(f"Room floor dimensions must be at least 3x3, got {width}x{height}")

        self._room_floor_w = width
        self._room_floor_h = height

        self._num_shared_generators = num_shared_generators
        self._num_altars = num_altars
        self._num_mines = num_mines

        self._agents_input = agents
        if isinstance(agents, int):
            if agents < 0:
                raise ValueError("Number of agents cannot be negative.")
            self._num_total_agents = agents
        elif isinstance(agents, (dict, DictConfig)):
            current_total_agents = 0
            for agent_name, count_val in agents.items():
                if not isinstance(count_val, int) or count_val < 0:
                    raise ValueError(
                        f"Agent count for '{str(agent_name)}' must be a non-negative integer, got {count_val}"
                    )
                current_total_agents += count_val
            self._num_total_agents = current_total_agents
        else:
            raise TypeError(f"Agents parameter must be an int, dict, or DictConfig, got {type(agents)}")

        if arrangement not in [None, "horizontal", "vertical"]:
            raise ValueError("Arrangement must be 'horizontal', 'vertical', or None for random.")

        self._arrangement = arrangement if arrangement else self._rng.choice(["horizontal", "vertical"])

        # Calculate core dimensions (floor spaces + their 1-thick surrounding walls + 1-thick shared wall)
        if self._arrangement == "horizontal":
            # fw + wall + shared_wall + wall + fw = 2*fw + 3 walls
            core_w = 2 * self._room_floor_w + 3
            core_h = self._room_floor_h + 2  # +2 for top/bottom walls
        else:  # vertical
            core_h = 2 * self._room_floor_h + 3
            core_w = self._room_floor_w + 2

        actual_grid_width = core_w + 2 * self._border_width
        actual_grid_height = core_h + 2 * self._border_width

        super().__init__(
            border_width=self._border_width, border_object="wall", labels=["two_rooms_coord", self._arrangement]
        )

        self._width = actual_grid_width
        self._height = actual_grid_height

        self._occ = np.zeros((self._height, self._width), dtype=bool)
        self.set_size_labels(self._width, self._height)

    def _get_empty_cells(self, grid: np.ndarray, floor_coords: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Helper to get currently empty cells from a list of floor coordinates."""
        empty = []
        for r_f, c_f in floor_coords:
            if 0 <= r_f < self._height and 0 <= c_f < self._width and grid[r_f, c_f] == "empty":
                empty.append((r_f, c_f))
        return empty

    def _place_objects_in_cells(
        self, grid: np.ndarray, object_name: str, count: int, available_cells: List[Tuple[int, int]]
    ):
        self._rng.shuffle(available_cells)
        placed_count = 0
        for i in range(min(count, len(available_cells))):
            r, c = available_cells[i]
            grid[r, c] = object_name
            self._occ[r, c] = True
            placed_count += 1

        del available_cells[:placed_count]

        if placed_count < count:
            print(f"Warning: Could only place {placed_count}/{count} of '{object_name}'.")

    def _build(self) -> np.ndarray:
        grid = np.full((self._height, self._width), "wall", dtype=object)
        self._occ.fill(True)

        r1_floor_coords_abs: List[Tuple[int, int]] = []
        r2_floor_coords_abs: List[Tuple[int, int]] = []
        shared_wall_coords_abs: List[Tuple[int, int]] = []

        # Define coordinates for cells adjacent to the shared wall within each room's floor space.
        # These are potential locations to exclude for item placement.
        adj_to_shared_wall_r1_coords: Set[Tuple[int, int]] = set()
        adj_to_shared_wall_r2_coords: Set[Tuple[int, int]] = set()

        if self._arrangement == "horizontal":
            r1_floor_tl_r_abs = self._border_width + 1
            r1_floor_tl_c_abs = self._border_width + 1
            for r_offset in range(self._room_floor_h):
                for c_offset in range(self._room_floor_w):
                    r1_floor_coords_abs.append((r1_floor_tl_r_abs + r_offset, r1_floor_tl_c_abs + c_offset))

            shared_wall_c_abs = self._border_width + 1 + self._room_floor_w
            # Shared wall length is the room floor height as rooms are same height in this setup
            shared_wall_start_r_abs = self._border_width + 1
            for r_offset in range(self._room_floor_h):
                shared_wall_coords_abs.append((shared_wall_start_r_abs + r_offset, shared_wall_c_abs))

            r2_floor_tl_r_abs = self._border_width + 1
            r2_floor_tl_c_abs = shared_wall_c_abs + 1
            for r_offset in range(self._room_floor_h):
                for c_offset in range(self._room_floor_w):
                    r2_floor_coords_abs.append((r2_floor_tl_r_abs + r_offset, r2_floor_tl_c_abs + c_offset))

            # Populate coordinates adjacent to the shared wall for horizontal arrangement
            adj_c_in_r1 = shared_wall_c_abs - 1
            adj_c_in_r2 = shared_wall_c_abs + 1
            for r_offset in range(self._room_floor_h):
                r_coord = shared_wall_start_r_abs + r_offset
                adj_to_shared_wall_r1_coords.add((r_coord, adj_c_in_r1))
                adj_to_shared_wall_r2_coords.add((r_coord, adj_c_in_r2))

        else:  # vertical
            r1_floor_tl_r_abs = self._border_width + 1
            r1_floor_tl_c_abs = self._border_width + 1
            for r_offset in range(self._room_floor_h):
                for c_offset in range(self._room_floor_w):
                    r1_floor_coords_abs.append((r1_floor_tl_r_abs + r_offset, r1_floor_tl_c_abs + c_offset))

            shared_wall_r_abs = self._border_width + 1 + self._room_floor_h
            # Shared wall length is the room floor width
            shared_wall_start_c_abs = self._border_width + 1
            for c_offset in range(self._room_floor_w):
                shared_wall_coords_abs.append((shared_wall_r_abs, shared_wall_start_c_abs + c_offset))

            r2_floor_tl_r_abs = shared_wall_r_abs + 1
            r2_floor_tl_c_abs = self._border_width + 1
            for r_offset in range(self._room_floor_h):
                for c_offset in range(self._room_floor_w):
                    r2_floor_coords_abs.append((r2_floor_tl_r_abs + r_offset, r2_floor_tl_c_abs + c_offset))

            # Populate coordinates adjacent to the shared wall for vertical arrangement
            adj_r_in_r1 = shared_wall_r_abs - 1
            adj_r_in_r2 = shared_wall_r_abs + 1
            for c_offset in range(self._room_floor_w):
                c_coord = shared_wall_start_c_abs + c_offset
                adj_to_shared_wall_r1_coords.add((adj_r_in_r1, c_coord))
                adj_to_shared_wall_r2_coords.add((adj_r_in_r2, c_coord))

        # Carve Room Floors
        for r_f, c_f in r1_floor_coords_abs:
            if 0 <= r_f < self._height and 0 <= c_f < self._width:
                grid[r_f, c_f] = "empty"
                self._occ[r_f, c_f] = False
        for r_f, c_f in r2_floor_coords_abs:
            if 0 <= r_f < self._height and 0 <= c_f < self._width:
                grid[r_f, c_f] = "empty"
                self._occ[r_f, c_f] = False

        # Place Shared Generators
        self._rng.shuffle(shared_wall_coords_abs)
        num_gen_placed = 0
        for i in range(min(self._num_shared_generators, len(shared_wall_coords_abs))):
            r_s, c_s = shared_wall_coords_abs[i]
            if 0 <= r_s < self._height and 0 <= c_s < self._width:  # Should always be true by construction
                grid[r_s, c_s] = "generator_red"
                num_gen_placed += 1
        if num_gen_placed < self._num_shared_generators:
            print(f"Warning: Could only place {num_gen_placed}/{self._num_shared_generators} shared generators.")

        # Designate Altar/Mine Rooms
        room_assignments = ["altar_room", "mine_room"]
        self._rng.shuffle(room_assignments)
        r1_designation, r2_designation = room_assignments[0], room_assignments[1]

        # Get all initially empty floor cells in each room
        initial_r1_empty_cells = self._get_empty_cells(grid, r1_floor_coords_abs)
        initial_r2_empty_cells = self._get_empty_cells(grid, r2_floor_coords_abs)

        # Filter out cells adjacent to the shared wall for item placement
        r1_eligible_for_items = [cell for cell in initial_r1_empty_cells if cell not in adj_to_shared_wall_r1_coords]
        r2_eligible_for_items = [cell for cell in initial_r2_empty_cells if cell not in adj_to_shared_wall_r2_coords]

        # Place Altars and Mines
        if r1_designation == "altar_room":
            self._place_objects_in_cells(grid, "altar", self._num_altars, r1_eligible_for_items)
        else:  # r1 is mine_room
            self._place_objects_in_cells(grid, "mine_red", self._num_mines, r1_eligible_for_items)

        if r2_designation == "altar_room":
            self._place_objects_in_cells(grid, "altar", self._num_altars, r2_eligible_for_items)
        else:  # r2 is mine_room
            self._place_objects_in_cells(grid, "mine_red", self._num_mines, r2_eligible_for_items)

        # Prepare agent symbols list
        agent_symbols_list: List[str] = []
        if isinstance(self._agents_input, int):
            agent_symbols_list = ["agent.agent"] * self._num_total_agents
        elif isinstance(self._agents_input, (dict, DictConfig)):
            temp_list = []
            for agent_name, count_val in self._agents_input.items():
                # Validation for count_val (must be int >= 0) already done in __init__
                # but good to be defensive if self._agents_input could be modified post-init.
                # For now, assume __init__ guarantees valid counts in DictConfig.
                temp_list.extend([f"agent.{str(agent_name)}"] * count_val)

            # Shuffle to mix agent types before placement
            self._rng.shuffle(temp_list)
            agent_symbols_list = temp_list
        # else: type error handled in __init__

        # Place Agents alternately
        # Get currently empty cells for agent placement (after items have been placed)
        r1_empty_for_agents = self._get_empty_cells(grid, r1_floor_coords_abs)
        r2_empty_for_agents = self._get_empty_cells(grid, r2_floor_coords_abs)

        self._rng.shuffle(r1_empty_for_agents)
        self._rng.shuffle(r2_empty_for_agents)

        agents_placed_count = 0
        room1_first = self._rng.choice([True, False])

        # Number of agents we will attempt to place based on the generated symbols list
        num_agents_to_attempt_placement = len(agent_symbols_list)

        for i in range(num_agents_to_attempt_placement):
            agent_symbol_to_place = agent_symbols_list[i]

            target_room1 = (room1_first and i % 2 == 0) or (not room1_first and i % 2 != 0)

            placed_in_iteration = False
            if target_room1 and r1_empty_for_agents:
                r_a, c_a = r1_empty_for_agents.pop(0)
                grid[r_a, c_a] = agent_symbol_to_place
                self._occ[r_a, c_a] = True
                placed_in_iteration = True
            elif not target_room1 and r2_empty_for_agents:
                r_a, c_a = r2_empty_for_agents.pop(0)
                grid[r_a, c_a] = agent_symbol_to_place
                self._occ[r_a, c_a] = True
                placed_in_iteration = True
            else:  # Target room is full, try the other room
                if r1_empty_for_agents:
                    r_a, c_a = r1_empty_for_agents.pop(0)
                    grid[r_a, c_a] = agent_symbol_to_place
                    self._occ[r_a, c_a] = True
                    placed_in_iteration = True
                elif r2_empty_for_agents:
                    r_a, c_a = r2_empty_for_agents.pop(0)
                    grid[r_a, c_a] = agent_symbol_to_place
                    self._occ[r_a, c_a] = True
                    placed_in_iteration = True

            if placed_in_iteration:
                agents_placed_count += 1
            else:
                # print(f"Warning: No empty space left to place agent {i+1} ('{agent_symbol_to_place}').")
                break  # Stop if no space found for current agent

        if agents_placed_count < num_agents_to_attempt_placement:
            print(
                f"Warning: Could only place {agents_placed_count}/{num_agents_to_attempt_placement} agents. "
                f"(Initial request for {self._num_total_agents} total agents from input configuration)."
            )
        # e.g. if DictConfig had invalid counts initially
        elif self._num_total_agents > num_agents_to_attempt_placement:
            print(
                f"Warning: Placed {agents_placed_count} agents. "
                f"Initial request for {self._num_total_agents} total agents, "
                f"but only {num_agents_to_attempt_placement} were validly specified."
            )

        return grid
