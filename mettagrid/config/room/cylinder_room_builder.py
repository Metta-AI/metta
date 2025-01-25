from typing import Set, Tuple
import numpy as np
from omegaconf import DictConfig

from mettagrid.config.room_builder import RoomBuilder, SYMBOLS

class CylinderRoomBuilder(RoomBuilder):
    def __init__(
        self,
        width: int,
        height: int,
        cylinder_params: DictConfig,
        num_agents: int = 1,
        seed = None,
        border_width: int = 0,
        border_object: str = SYMBOLS["wall"],
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._width = width
        self._height = height
        self._cylinder_params = cylinder_params
        self._num_agents = num_agents
        self._rng = np.random.default_rng(seed)

        # Validate inputs
        assert 3 <= cylinder_params['length'], "Cylinder length must be at least 3"
        assert cylinder_params['orientation'] in ['horizontal', 'vertical'], "Invalid orientation"

        # Initialize grid
        self._grid = np.full((self._height, self._width), SYMBOLS["empty"], dtype=str)
        self._cylinder_positions = set()

    def _build(self) -> np.ndarray:
        if self._cylinder_params['orientation'] == 'horizontal':
            self.place_horizontal_cylinder()
        else:
            self.place_vertical_cylinder()

        valid_positions = self._get_valid_positions()
        self._place_elements(valid_positions)

        return self._grid

    def place_horizontal_cylinder(self) -> None:
        """Place a horizontal cylinder with generator and agents."""
        center_y = self._height // 2
        wall_length = self._cylinder_params['length']
        start_x = (self._width - wall_length) // 2

        # Create parallel walls
        for x in range(start_x, start_x + wall_length):
            if not self._cylinder_params['both_ends'] or (x != start_x and x != start_x + wall_length - 1):
                self._grid[center_y - 1, x] = SYMBOLS["wall"]  # Top wall
                self._grid[center_y + 1, x] = SYMBOLS["wall"]  # Bottom wall
                self._cylinder_positions.update({(x, center_y - 1), (x, center_y + 1)})

        # Place generator
        generator_x = start_x + (wall_length // 2)
        self._grid[center_y, generator_x] = SYMBOLS["generator"]
        self._cylinder_positions.add((generator_x, center_y))

        # Place agents
        agent_start_x = start_x + (wall_length - self._num_agents) // 2
        for i in range(self._num_agents):
            self._grid[center_y - 2, agent_start_x + i] = SYMBOLS["agent"]
            self._cylinder_positions.add((agent_start_x + i, center_y - 2))

    def place_vertical_cylinder(self) -> None:
        """Place a vertical cylinder with generator and agents."""
        center_x = self._width // 2
        wall_length = self._cylinder_params['length']
        start_y = (self._height - wall_length) // 2

        # Create parallel walls
        for y in range(start_y, start_y + wall_length):
            if not self._cylinder_params['both_ends'] or (y != start_y and y != start_y + wall_length - 1):
                self._grid[y, center_x - 1] = SYMBOLS["wall"]  # Left wall
                self._grid[y, center_x + 1] = SYMBOLS["wall"]  # Right wall
                self._cylinder_positions.update({(center_x - 1, y), (center_x + 1, y)})

        # Place generator
        generator_y = start_y + (wall_length // 2)
        self._grid[generator_y, center_x] = SYMBOLS["generator"]
        self._cylinder_positions.add((center_x, generator_y))

        # Place agents
        agent_start_y = start_y + (wall_length - self._num_agents) // 2
        for i in range(self._num_agents):
            self._grid[agent_start_y + i, center_x - 2] = SYMBOLS["agent"]
            self._cylinder_positions.add((center_x - 2, agent_start_y + i))

    def _get_valid_positions(self) -> Set[Tuple[int, int]]:
        """Get positions that aren't part of the cylinder structure."""
        valid_positions = set()
        for x in range(1, self._width-1):
            for y in range(1, self._height-1):
                if (x, y) not in self._cylinder_positions:
                    valid_positions.add((x, y))
        return valid_positions

    def _place_elements(self, valid_positions: Set[Tuple[int, int]]) -> None:
        """Place altar and converter in valid positions."""
        if self._cylinder_params['orientation'] == 'horizontal':
            top_positions = [(x, y) for x, y in valid_positions if y < self._height//2]
            left_positions = [pos for pos in top_positions if pos[0] < self._width//2]
            right_positions = [pos for pos in top_positions if pos[0] >= self._width//2]
        else:
            left_positions = [pos for pos in valid_positions if pos[0] < self._width//2]
            right_positions = [pos for pos in valid_positions if pos[0] >= self._width//2]

        if left_positions and right_positions:
            altar_pos = self._rng.choice(left_positions)
            converter_pos = self._rng.choice(right_positions)
            self._grid[altar_pos[1], altar_pos[0]] = SYMBOLS["altar"]
            self._grid[converter_pos[1], converter_pos[0]] = SYMBOLS["converter"]
