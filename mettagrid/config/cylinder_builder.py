from mettagrid.config.room_builder import MettaGridRoomBuilder
import numpy as np
import random
from typing import Set, Tuple


class CylinderRoom(MettaGridRoomBuilder):
    EMPTY, WALL, AGENT, GENERATOR = ' ', 'W', 'A', 'g'
    HEART, CONVERTER = 'a', 'c'

    def __init__(self, width, height, cylinder_params, num_agents=1, seed=None):
        self.width = width
        self.height = height
        self.cylinder_params = cylinder_params
        self.num_agents = num_agents
        self.seed = seed

        # Validate inputs
        assert 3 <= cylinder_params['length'], "Cylinder length must be at least 3"
        assert cylinder_params['orientation'] in ['horizontal', 'vertical'], "Invalid orientation"
        
        # Initialize grid
        self.grid = np.full((height, width), self.EMPTY, dtype=str)
        self.cylinder_positions = set()
    
    def build_room(self):
        return self.create_cylinder() 

    def add_border_walls(self) -> None:
        """Add border walls to the grid."""
        self.grid[0, :] = self.WALL
        self.grid[-1, :] = self.WALL
        self.grid[:, 0] = self.WALL
        self.grid[:, -1] = self.WALL

    def place_horizontal_cylinder(self) -> None:
        """Place a horizontal cylinder with generator and agents."""
        center_y = self.height // 2
        wall_length = self.cylinder_params['length']
        start_x = (self.width - wall_length) // 2
        
        # Create parallel walls
        for x in range(start_x, start_x + wall_length):
            if not self.cylinder_params['both_ends'] or (x != start_x and x != start_x + wall_length - 1):
                self.grid[center_y - 1, x] = self.WALL  # Top wall
                self.grid[center_y + 1, x] = self.WALL  # Bottom wall
                self.cylinder_positions.update({(x, center_y - 1), (x, center_y + 1)})
        
        # Place generator
        generator_x = start_x + (wall_length // 2)
        self.grid[center_y, generator_x] = self.GENERATOR
        self.cylinder_positions.add((generator_x, center_y))
        
        # Place agents
        agent_start_x = start_x + (wall_length - self.num_agents) // 2
        for i in range(self.num_agents):
            self.grid[center_y - 2, agent_start_x + i] = self.AGENT
            self.cylinder_positions.add((agent_start_x + i, center_y - 2))

    def place_vertical_cylinder(self) -> None:
        """Place a vertical cylinder with generator and agents."""
        center_x = self.width // 2
        wall_length = self.cylinder_params['length']
        start_y = (self.height - wall_length) // 2
        
        # Create parallel walls
        for y in range(start_y, start_y + wall_length):
            if not self.cylinder_params['both_ends'] or (y != start_y and y != start_y + wall_length - 1):
                self.grid[y, center_x - 1] = self.WALL  # Left wall
                self.grid[y, center_x + 1] = self.WALL  # Right wall
                self.cylinder_positions.update({(center_x - 1, y), (center_x + 1, y)})
        
        # Place generator
        generator_y = start_y + (wall_length // 2)
        self.grid[generator_y, center_x] = self.GENERATOR
        self.cylinder_positions.add((center_x, generator_y))
        
        # Place agents
        agent_start_y = start_y + (wall_length - self.num_agents) // 2
        for i in range(self.num_agents):
            self.grid[agent_start_y + i, center_x - 2] = self.AGENT
            self.cylinder_positions.add((center_x - 2, agent_start_y + i))

    def get_valid_positions(self) -> Set[Tuple[int, int]]:
        """Get positions that aren't part of the cylinder structure."""
        valid_positions = set()
        for x in range(1, self.width-1):
            for y in range(1, self.height-1):
                if (x, y) not in self.cylinder_positions:
                    valid_positions.add((x, y))
        return valid_positions

    def place_elements(self, valid_positions: Set[Tuple[int, int]]) -> None:
        """Place heart altar and converter in valid positions."""
        if self.cylinder_params['orientation'] == 'horizontal':
            top_positions = [(x, y) for x, y in valid_positions if y < self.height//2]
            left_positions = [pos for pos in top_positions if pos[0] < self.width//2]
            right_positions = [pos for pos in top_positions if pos[0] >= self.width//2]
        else:
            left_positions = [pos for pos in valid_positions if pos[0] < self.width//2]
            right_positions = [pos for pos in valid_positions if pos[0] >= self.width//2]
        
        if left_positions and right_positions:
            altar_pos = random.choice(left_positions)
            converter_pos = random.choice(right_positions)
            self.grid[altar_pos[1], altar_pos[0]] = self.HEART
            self.grid[converter_pos[1], converter_pos[0]] = self.CONVERTER

    def create_cylinder(self) -> np.ndarray:
        """Generate and return the cylinder environment."""
        self.add_border_walls()
        
        if self.cylinder_params['orientation'] == 'horizontal':
            self.place_horizontal_cylinder()
        else:
            self.place_vertical_cylinder()
        
        valid_positions = self.get_valid_positions()
        self.place_elements(valid_positions)
        
        return self.grid