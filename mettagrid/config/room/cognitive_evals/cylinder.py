from typing import Set, Tuple
import numpy as np
import random
from omegaconf import DictConfig

from mettagrid.config.room.room import Room

class Cylinder(Room):
    def __init__(self, width: int, height: int, cylinder_params: DictConfig,
                 agents: int | DictConfig = 1, border_width: int = 1, border_object: str = "wall"):
        super().__init__(border_width=border_width, border_object=border_object)
        self._width = width
        self._height = height
        self._cylinder_params = cylinder_params
        self._agents = agents
        assert cylinder_params['length'] >= 3, "Cylinder length must be at least 3"

    def _build(self) -> np.ndarray:
        self._grid = np.full((self._height, self._width), "empty", dtype='<U50')
        self._cylinder_positions = set()
        if self._cylinder_params.horizontal:
            self.place_horizontal_cylinder()
        else:
            self.place_vertical_cylinder()
        valid_positions = {
            (x, y) for x in range(1, self._width - 1)
            for y in range(1, self._height - 1)
            if (x, y) not in self._cylinder_positions
        }
        return self._place_elements(valid_positions)

    def place_horizontal_cylinder(self) -> None:
        center_y = self._height // 2
        wall_length = self._cylinder_params['length']
        start_x = (self._width - wall_length) // 2

        for x in range(start_x, start_x + wall_length):
            if not self._cylinder_params['both_ends'] or (x != start_x and x != start_x + wall_length - 1):
                self._grid[center_y - 1, x] = "wall"
                self._grid[center_y + 1, x] = "wall"
                self._cylinder_positions.update({(x, center_y - 1), (x, center_y + 1)})

        mine_x = start_x + wall_length // 2
        self._grid[center_y, mine_x] = "mine"
        self._cylinder_positions.add((mine_x, center_y))

        agent_start_x = start_x + (wall_length - self._agents) // 2
        for i in range(self._agents):
            self._grid[center_y - 2, agent_start_x + i] = "agent.agent"
            self._cylinder_positions.add((agent_start_x + i, center_y - 2))

    def place_vertical_cylinder(self) -> None:
        center_x = self._width // 2
        wall_length = self._cylinder_params['length']
        start_y = (self._height - wall_length) // 2

        for y in range(start_y, start_y + wall_length):
            if not self._cylinder_params['both_ends'] or (y != start_y and y != start_y + wall_length - 1):
                self._grid[y, center_x - 1] = "wall"
                self._grid[y, center_x + 1] = "wall"
                self._cylinder_positions.update({(center_x - 1, y), (center_x + 1, y)})

        mine_y = start_y + wall_length // 2
        self._grid[mine_y, center_x] = "mine"
        self._cylinder_positions.add((center_x, mine_y))

        agent_start_y = start_y + (wall_length - self._agents) // 2
        for i in range(self._agents):
            self._grid[agent_start_y + i, center_x - 2] = "agent.agent"
            self._cylinder_positions.add((center_x - 2, agent_start_y + i))

    def _place_elements(self, valid_positions: Set[Tuple[int, int]]) -> np.ndarray:
        new_grid = self._grid.copy()
        if self._cylinder_params.horizontal:
            left_positions = [pos for pos in valid_positions if pos[0] < self._width // 2 and pos[1] < self._height // 2]
            right_positions = [pos for pos in valid_positions if pos[0] >= self._width // 2 and pos[1] < self._height // 2]
        else:
            left_positions = [pos for pos in valid_positions if pos[0] < self._width // 2]
            right_positions = [pos for pos in valid_positions if pos[0] >= self._width // 2]

        if left_positions and right_positions:
            altar_pos = random.choice(left_positions)
            generator_pos = random.choice(right_positions)
            new_grid[altar_pos[1], altar_pos[0]] = "altar"
            new_grid[generator_pos[1], generator_pos[0]] = "generator"
        return new_grid
