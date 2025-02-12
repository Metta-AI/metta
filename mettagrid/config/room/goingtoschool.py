from typing import Set, Tuple
import numpy as np
from omegaconf import DictConfig

# Assuming Room is imported from the mettagrid framework.
from mettagrid.config.room.room import Room

class Cylinder(Room):
    """
    A cylinder room for the 'going to school' environment.
    
    Each room contains:
      - A centered generator (placed in the middle of the cylinder wall)
      - One agent (placed once per room)
      - A heart altar and a convertor placed at random valid positions around the cylinder.
    
    The room dimensions and cylinder parameters (such as wall length,
    orientation, and whether to have openings at both ends) are passed via the YAML config.
    """
    def __init__(
        self,
        width: int,
        height: int,
        cylinder_params: DictConfig,
        agents: int = 1,
        seed: int | None = None,
        border_width: int = 0,
        border_object: str = "wall",
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._width = width
        self._height = height
        self._cylinder_params = cylinder_params
        self._agents = agents  # For goingtoschool, always 1 per room.
        self._rng = np.random.default_rng(seed)
        
        # Ensure cylinder length is at least 3.
        assert cylinder_params['length'] >= 3, "Cylinder length must be at least 3"
        
        # Initialize the grid with "empty" cells.
        self._grid = np.full((self._height, self._width), "empty", dtype='<U50')
        self._cylinder_positions = set()

    def _build(self) -> np.ndarray:
        # Build the cylinder based on its orientation.
        if self._cylinder_params.horizontal:
            self.place_horizontal_cylinder()
        else:
            self.place_vertical_cylinder()

        valid_positions = self._get_valid_positions()
        self._place_elements(valid_positions)

        return self._grid

    def place_horizontal_cylinder(self) -> None:
        center_y = self._height // 2
        wall_length = self._cylinder_params['length']
        start_x = (self._width - wall_length) // 2

        # Build the parallel walls of the cylinder.
        for x in range(start_x, start_x + wall_length):
            # If 'both_ends' is False, skip the wall at the extreme ends.
            if not self._cylinder_params['both_ends'] or (x != start_x and x != start_x + wall_length - 1):
                self._grid[center_y - 1, x] = "wall"
                self._grid[center_y + 1, x] = "wall"
                self._cylinder_positions.update({(x, center_y - 1), (x, center_y + 1)})

        # Place the generator exactly in the middle of the cylinder.
        generator_x = start_x + (wall_length // 2)
        self._grid[center_y, generator_x] = "generator"
        self._cylinder_positions.add((generator_x, center_y))

        # Place the single agent for this room.
        # (For simplicity, we place the agent a couple cells above the generator.)
        agent_x = generator_x
        self._grid[center_y - 2, agent_x] = "agent"
        self._cylinder_positions.add((agent_x, center_y - 2))

    def place_vertical_cylinder(self) -> None:
        center_x = self._width // 2
        wall_length = self._cylinder_params['length']
        start_y = (self._height - wall_length) // 2

        # Build the parallel walls of the cylinder.
        for y in range(start_y, start_y + wall_length):
            if not self._cylinder_params['both_ends'] or (y != start_y and y != start_y + wall_length - 1):
                self._grid[y, center_x - 1] = "wall"
                self._grid[y, center_x + 1] = "wall"
                self._cylinder_positions.update({(center_x - 1, y), (center_x + 1, y)})

        # Place the generator in the middle of the cylinder.
        generator_y = start_y + (wall_length // 2)
        self._grid[generator_y, center_x] = "generator"
        self._cylinder_positions.add((center_x, generator_y))

        # Place the single agent for this room.
        # (For simplicity, we place the agent a couple cells to the left of the generator.)
        agent_y = generator_y
        self._grid[agent_y, center_x - 2] = "agent"
        self._cylinder_positions.add((center_x - 2, agent_y))

    def _get_valid_positions(self) -> Set[Tuple[int, int]]:
        """
        Compute the set of grid positions that are not part of the cylinder structure.
        """
        valid_positions = set()
        for x in range(1, self._width - 1):
            for y in range(1, self._height - 1):
                if (x, y) not in self._cylinder_positions:
                    valid_positions.add((x, y))
        return valid_positions

    def _place_elements(self, valid_positions: Set[Tuple[int, int]]) -> None:
        """
        Place a heart altar and a convertor at random valid positions.
        For horizontal cylinders, positions are chosen from the top half (split into left/right halves);
        for vertical cylinders, the valid positions are split by the x-axis.
        """
        if self._cylinder_params.horizontal:
            top_positions = [(x, y) for (x, y) in valid_positions if y < self._height // 2]
            left_positions = [pos for pos in top_positions if pos[0] < self._width // 2]
            right_positions = [pos for pos in top_positions if pos[0] >= self._width // 2]
        else:
            left_positions = [pos for pos in valid_positions if pos[0] < self._width // 2]
            right_positions = [pos for pos in valid_positions if pos[0] >= self._width // 2]

        if left_positions:
            heart_altar_pos = self._rng.choice(left_positions)
            self._grid[heart_altar_pos[1], heart_altar_pos[0]] = "heart_altar"
        if right_positions:
            convertor_pos = self._rng.choice(right_positions)
            self._grid[convertor_pos[1], convertor_pos[0]] = "convertor"
