from typing import Set, Tuple
import numpy as np
from omegaconf import DictConfig

# Assuming that Room is defined in mettagrid.config.room.room.
from mettagrid.config.room.room import Room

class RoomWithinRoomWithinRoom(Room):
    def __init__(
        self,
        width: int,
        height: int,
        middle_size_min: int,
        middle_size_max: int,
        middle_room_gap_min: int,
        middle_room_gap_max: int,
        inner_size_min: int,
        inner_size_max: int,
        inner_room_gap_min: int,
        inner_room_gap_max: int,
        border_width: int = 1,
        border_object: str = "wall",
        agents: int | DictConfig = 1,
        seed=None,
    ):
        """
        Creates an environment with three nested rooms.
        
        - The outer room is the overall grid, bounded by a solid wall.
        - The middle room is centered inside the outer room and has its own walls,
          with a door gap placed along one randomly chosen wall.
        - The innermost room is centered within the middle room. It, too, has its
          own walls with a door gap randomly placed along one wall. Inside the
          innermost room:
            - Top left: converter
            - Top right: altar
            - Bottom right: generator
            - Bottom left: agent
          
        In addition, a cluster of outer room objects (converter, altar, generator)
        is placed together in one randomly chosen corner of the overall grid.
        """
        super().__init__(border_width=border_width, border_object=border_object)
        self._overall_width = width
        self._overall_height = height
        self._border_width = border_width
        self._border_object = border_object
        
        # Middle room parameters
        self._middle_size_min = middle_size_min
        self._middle_size_max = middle_size_max
        self._middle_room_gap_min = middle_room_gap_min
        self._middle_room_gap_max = middle_room_gap_max

        # Innermost room parameters
        self._inner_size_min = inner_size_min
        self._inner_size_max = inner_size_max
        self._inner_room_gap_min = inner_room_gap_min
        self._inner_room_gap_max = inner_room_gap_max

        self._agents = agents if isinstance(agents, int) else agents
        self._rng = np.random.default_rng(seed)
        self._grid = np.full((self._overall_height, self._overall_width), "empty", dtype='<U50')
        self._wall_positions: Set[Tuple[int, int]] = set()

        # Sample dimensions for the middle room.
        self._middle_width = self._rng.integers(self._middle_size_min, self._middle_size_max + 1)
        self._middle_height = self._rng.integers(self._middle_size_min, self._middle_size_max + 1)
        # Sample dimensions for the innermost room.
        self._inner_width = self._rng.integers(self._inner_size_min, self._inner_size_max + 1)
        self._inner_height = self._rng.integers(self._inner_size_min, self._inner_size_max + 1)

    def _build(self) -> np.ndarray:
        # --- Draw the outer room walls (around the entire grid) ---
        for y in range(self._overall_height):
            for x in range(self._overall_width):
                if (
                    x < self._border_width or x >= self._overall_width - self._border_width or 
                    y < self._border_width or y >= self._overall_height - self._border_width
                ):
                    self._grid[y, x] = self._border_object
                    self._wall_positions.add((x, y))
                    
        # --- Build the middle room (centered within the outer room) ---
        middle_width = self._middle_width
        middle_height = self._middle_height

        middle_left = (self._overall_width - middle_width) // 2
        middle_top = (self._overall_height - middle_height) // 2
        middle_right = middle_left + middle_width - 1
        middle_bottom = middle_top + middle_height - 1

        # Randomly select which wall will have the door gap for the middle room.
        middle_door_wall = self._rng.choice(["top", "bottom", "left", "right"])
        if middle_door_wall in ["top", "bottom"]:
            middle_door_gap = self._rng.integers(self._middle_room_gap_min, self._middle_room_gap_max + 1)
            max_middle_gap = middle_width - 2 * self._border_width - 2
            if max_middle_gap < middle_door_gap:
                middle_door_gap = max_middle_gap if max_middle_gap > 0 else middle_door_gap
            door_start_middle = middle_left + self._border_width + ((middle_width - 2 * self._border_width - middle_door_gap) // 2)
        else:
            middle_door_gap = self._rng.integers(self._middle_room_gap_min, self._middle_room_gap_max + 1)
            max_middle_gap = middle_height - 2 * self._border_width - 2
            if max_middle_gap < middle_door_gap:
                middle_door_gap = max_middle_gap if max_middle_gap > 0 else middle_door_gap
            door_start_middle = middle_top + self._border_width + ((middle_height - 2 * self._border_width - middle_door_gap) // 2)

        # Draw the middle room walls.
        for y in range(middle_top, middle_bottom + 1):
            for x in range(middle_left, middle_right + 1):
                if x == middle_left or x == middle_right or y == middle_top or y == middle_bottom:
                    if middle_door_wall == "top" and y == middle_top and door_start_middle <= x < door_start_middle + middle_door_gap:
                        self._grid[y, x] = "door"
                    elif middle_door_wall == "bottom" and y == middle_bottom and door_start_middle <= x < door_start_middle + middle_door_gap:
                        self._grid[y, x] = "door"
                    elif middle_door_wall == "left" and x == middle_left and door_start_middle <= y < door_start_middle + middle_door_gap:
                        self._grid[y, x] = "door"
                    elif middle_door_wall == "right" and x == middle_right and door_start_middle <= y < door_start_middle + middle_door_gap:
                        self._grid[y, x] = "door"
                    else:
                        self._grid[y, x] = self._border_object
                        self._wall_positions.add((x, y))

        # --- Build the innermost room (centered within the middle room) ---
        inner_width = self._inner_width
        inner_height = self._inner_height

        inner_left = middle_left + (middle_width - inner_width) // 2
        inner_top = middle_top + (middle_height - inner_height) // 2
        inner_right = inner_left + inner_width - 1
        inner_bottom = inner_top + inner_height - 1

        # Randomly select which wall will have the door gap for the innermost room.
        inner_door_wall = self._rng.choice(["top", "bottom", "left", "right"])
        if inner_door_wall in ["top", "bottom"]:
            inner_door_gap = self._rng.integers(self._inner_room_gap_min, self._inner_room_gap_max + 1)
            max_inner_gap = inner_width - 2 * self._border_width - 2
            if max_inner_gap < inner_door_gap:
                inner_door_gap = max_inner_gap if max_inner_gap > 0 else inner_door_gap
            door_start_inner = inner_left + self._border_width + ((inner_width - 2 * self._border_width - inner_door_gap) // 2)
        else:
            inner_door_gap = self._rng.integers(self._inner_room_gap_min, self._inner_room_gap_max + 1)
            max_inner_gap = inner_height - 2 * self._border_width - 2
            if max_inner_gap < inner_door_gap:
                inner_door_gap = max_inner_gap if max_inner_gap > 0 else inner_door_gap
            door_start_inner = inner_top + self._border_width + ((inner_height - 2 * self._border_width - inner_door_gap) // 2)

        # Draw the innermost room walls.
        for y in range(inner_top, inner_bottom + 1):
            for x in range(inner_left, inner_right + 1):
                if x == inner_left or x == inner_right or y == inner_top or y == inner_bottom:
                    if inner_door_wall == "top" and y == inner_top and door_start_inner <= x < door_start_inner + inner_door_gap:
                        self._grid[y, x] = "door"
                    elif inner_door_wall == "bottom" and y == inner_bottom and door_start_inner <= x < door_start_inner + inner_door_gap:
                        self._grid[y, x] = "door"
                    elif inner_door_wall == "left" and x == inner_left and door_start_inner <= y < door_start_inner + inner_door_gap:
                        self._grid[y, x] = "door"
                    elif inner_door_wall == "right" and x == inner_right and door_start_inner <= y < door_start_inner + inner_door_gap:
                        self._grid[y, x] = "door"
                    else:
                        self._grid[y, x] = self._border_object
                        self._wall_positions.add((x, y))

        # --- Place objects inside the innermost room (in its four corners) ---
        # Top left: converter
        inner_converter_pos = (inner_left + self._border_width, inner_top + self._border_width)
        self._grid[inner_converter_pos[1], inner_converter_pos[0]] = "converter"

        # Top right: altar
        inner_altar_pos = (inner_right - self._border_width, inner_top + self._border_width)
        self._grid[inner_altar_pos[1], inner_altar_pos[0]] = "altar"

        # Bottom right: generator
        inner_generator_pos = (inner_right - self._border_width, inner_bottom - self._border_width)
        self._grid[inner_generator_pos[1], inner_generator_pos[0]] = "generator"

        # Bottom left: agent
        inner_agent_pos = (inner_left + self._border_width, inner_bottom - self._border_width)
        self._grid[inner_agent_pos[1], inner_agent_pos[0]] = "agent.agent"

        # --- Place outer room objects as a grouped cluster in a random corner of the overall grid ---
        corners = ["top_left", "top_right", "bottom_left", "bottom_right"]
        selected_corner = self._rng.choice(corners)
        
        if selected_corner == "top_left":
            outer_obj_x = self._border_width + 1
            outer_obj_y = self._border_width + 1
            # Place items: converter at (x,y), altar to its right, generator below.
            self._grid[outer_obj_y, outer_obj_x] = "converter"
            if outer_obj_x + 1 < self._overall_width - self._border_width:
                self._grid[outer_obj_y, outer_obj_x + 1] = "altar"
            if outer_obj_y + 1 < self._overall_height - self._border_width:
                self._grid[outer_obj_y + 1, outer_obj_x] = "generator"
        elif selected_corner == "top_right":
            outer_obj_x = self._overall_width - self._border_width - 2
            outer_obj_y = self._border_width + 1
            # Place items: converter at (x,y), altar to its left, generator below.
            self._grid[outer_obj_y, outer_obj_x] = "converter"
            if outer_obj_x - 1 >= self._border_width:
                self._grid[outer_obj_y, outer_obj_x - 1] = "altar"
            if outer_obj_y + 1 < self._overall_height - self._border_width:
                self._grid[outer_obj_y + 1, outer_obj_x] = "generator"
        elif selected_corner == "bottom_left":
            outer_obj_x = self._border_width + 1
            outer_obj_y = self._overall_height - self._border_width - 2
            # Place items: converter at (x,y), altar to its right, generator above.
            self._grid[outer_obj_y, outer_obj_x] = "converter"
            if outer_obj_x + 1 < self._overall_width - self._border_width:
                self._grid[outer_obj_y, outer_obj_x + 1] = "altar"
            if outer_obj_y - 1 >= self._border_width:
                self._grid[outer_obj_y - 1, outer_obj_x] = "generator"
        elif selected_corner == "bottom_right":
            outer_obj_x = self._overall_width - self._border_width - 2
            outer_obj_y = self._overall_height - self._border_width - 2
            # Place items: converter at (x,y), altar to its left, generator above.
            self._grid[outer_obj_y, outer_obj_x] = "converter"
            if outer_obj_x - 1 >= self._border_width:
                self._grid[outer_obj_y, outer_obj_x - 1] = "altar"
            if outer_obj_y - 1 >= self._border_width:
                self._grid[outer_obj_y - 1, outer_obj_x] = "generator"

        return self._grid
