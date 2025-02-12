from typing import Set, Tuple, Union
import numpy as np
from omegaconf import DictConfig

from mettagrid.config.room.room import Room


class YMaze(Room):
    """
    A parameterizable room (a.k.a. YMaze or DividedRoom) that is long and thin (vertical).
    In the top half a vertical divider (the "stem") splits the room into two segments.
    In the left segment a converter is placed and in the right segment a generator.
    The heart altar is placed at the bottom end of the room.
    
    When used as a sub-room in a multi-room environment,
    this class always places exactly one agent.
    """

    def __init__(
        self,
        width: int,
        height: int,
        ymaze_params: DictConfig,
        agents: Union[int, DictConfig] = 1,
        seed: int = None,
        border_width: int = 0,
        border_object: str = "wall",
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._width = width
        self._height = height
        self._ymaze_params = ymaze_params

        # Force exactly one agent per sub-room.
        self._agents = 1
        
        self._border_width = border_width
        self._border_object = border_object
        self._rng = np.random.default_rng(seed)

        # Store sensory_range from the YAML (defaulting to 4) for future use if needed.
        self._sensory_range = self._ymaze_params.get('sensory_range', 4)

        # Initialize grid (rows, columns) with "empty" cells.
        self._grid = np.full((self._height, self._width), "empty", dtype='<U50')
        # Keep track of all positions occupied by divider and objects.
        self._occupied_positions: Set[Tuple[int, int]] = set()

    def _build(self) -> np.ndarray:
        # Build the room step by step.
        self.place_vertical_divider()
        self.place_objects()
        self.place_agents()
        return self._grid

    def place_vertical_divider(self) -> None:
        """
        Place a vertical divider (stem) in the top portion of the room.
        The divider is placed in the center column and extends for a length
        determined by 'stem_length'. Its thickness is parameterizable (default 1).
        """
        center_x = self._width // 2

        # Use 'stem_length' from ymaze_params; default to half the room’s height.
        divider_length = self._ymaze_params.get('stem_length', self._height // 2)
        assert divider_length >= 3, "Stem length must be at least 3"

        # Divider starts at the top (taking into account any border)
        start_y = self._border_width
        end_y = min(start_y + divider_length, self._height - self._border_width)

        # Divider thickness (number of columns) with default value 1.
        thickness = self._ymaze_params.get('thickness', 1)

        # Place divider cells.
        for offset in range(thickness):
            col = center_x - offset
            if col < self._border_width:
                break  # Do not write into the border area.
            for y in range(start_y, end_y):
                self._grid[y, col] = self._border_object
                self._occupied_positions.add((col, y))

    def place_objects(self) -> None:
        """
        Place the converter and generator in the top (divided) portion and
        the heart altar at the bottom of the room.
        
        In this design:
          - The top row (inside the border) is used for the divided segments.
          - The left segment (to the left of center) gets the converter.
          - The right segment (to the right of center) gets the generator.
          - The heart altar is centered at the bottom.
        """
        center_x = self._width // 2

        # Determine horizontal boundaries (avoiding border cells).
        left_bound = self._border_width
        right_bound = self._width - self._border_width - 1

        # Use the very top row (inside the border).
        top_row = self._border_width

        # Use 'arm_length' from ymaze_params to position the objects.
        arm_length = self._ymaze_params.get('arm_length', 4)
        converter_x = max(left_bound, center_x - arm_length)
        generator_x = min(right_bound, center_x + arm_length)

        # Place converter in the left top segment.
        self._grid[top_row, converter_x] = "converter"
        self._occupied_positions.add((converter_x, top_row))

        # Place generator in the right top segment.
        self._grid[top_row, generator_x] = "generator"
        self._occupied_positions.add((generator_x, top_row))

        # Place the heart altar at the bottom center.
        bottom_row = self._height - self._border_width - 1
        altar_x = center_x
        self._grid[bottom_row, altar_x] = "heart_altar"
        self._occupied_positions.add((altar_x, bottom_row))

    def place_agents(self) -> None:
        """
        Place exactly one agent in this room.
        We choose the row immediately above the heart altar.
        If that cell is already occupied, a simple leftward search is performed.
        """
        bottom_row = self._height - self._border_width - 1
        agent_row = bottom_row - 1 if bottom_row - 1 >= self._border_width else bottom_row

        # Since we force one agent per room:
        num_agents = self._agents  # This will be 1.
        agent_start_x = (self._width - num_agents) // 2

        # Place the single agent.
        candidate = agent_start_x
        while candidate >= self._border_width and self._grid[agent_row, candidate] != "empty":
            candidate -= 1
        self._grid[agent_row, candidate] = "agent.agent"
        self._occupied_positions.add((candidate, agent_row))

    def _get_valid_positions(self) -> Set[Tuple[int, int]]:
        """
        Return all positions (excluding border cells) that are still empty.
        (Useful if further random placements are desired.)
        """
        valid_positions = set()
        for y in range(self._border_width, self._height - self._border_width):
            for x in range(self._border_width, self._width - self._border_width):
                if (x, y) not in self._occupied_positions:
                    valid_positions.add((x, y))
        return valid_positions
