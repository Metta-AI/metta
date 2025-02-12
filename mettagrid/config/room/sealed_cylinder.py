from typing import Set, Tuple
import numpy as np
from omegaconf import DictConfig

from mettagrid.config.room.room import Room

class SealedCylinder(Room):
    def __init__(
        self,
        width: int,
        height: int,
        cylinder_params: DictConfig,
        agents: int | DictConfig = 1,
        seed=None,
        border_width: int = 1,
        border_object: str = "wall",
    ):
        """
        Creates a single-room environment containing three sealed cylinders.
        
        The overall grid (of size height x width) is divided horizontally into three regions.
        In each region a cylinder is drawn with the following properties:
          - Its dimensions (length and cylinder_width) are specified in cylinder_params.
          - One end is sealed (filled with wall cells), and the opposite end features a door.
          - A special resource is placed at a random interior cell.
            The left‐most cylinder gets a "heart altar", the middle one a "generator", and
            the right‐most a "convertor".
        
        The agent is placed just outside the door of the first (left‐most) cylinder.
        """
        super().__init__(border_width=border_width, border_object=border_object)
        self._overall_width = width
        self._overall_height = height
        self._cylinder_params = cylinder_params
        self._agents = agents if isinstance(agents, int) else agents
        self._border_width = border_width
        self._rng = np.random.default_rng(seed)
        self._grid = np.full((self._overall_height, self._overall_width), "empty", dtype='<U50')
        self._wall_positions: Set[Tuple[int, int]] = set()

    def _build(self) -> np.ndarray:
        # List of special resources for the three cylinders.
        special_items = ["heart altar", "generator", "convertor"]
        num_cylinders = len(special_items)
        # Divide the overall grid width into equal regions.
        region_width = self._overall_width // num_cylinders

        # Extract common cylinder parameters.
        cyl_length = self._cylinder_params["length"]
        cyl_width = self._cylinder_params["cylinder_width"]
        # Determine which end is sealed (e.g., "left" means the left side is sealed).
        sealed_end = self._cylinder_params.get("sealed_end", "left")

        for i, special_item in enumerate(special_items):
            # Define the horizontal region for cylinder i.
            region_start_x = i * region_width
            region_end_x = (i + 1) * region_width - 1
            region_center_x = (region_start_x + region_end_x) // 2
            region_center_y = self._overall_height // 2

            # Center the cylinder within its region.
            cyl_start_x = region_center_x - (cyl_length // 2)
            cyl_start_y = region_center_y - (cyl_width // 2)
            cyl_end_x = cyl_start_x + cyl_length - 1
            cyl_end_y = cyl_start_y + cyl_width - 1

            # Determine door side and coordinates.
            # The door is on the side opposite to the sealed end.
            if sealed_end == "left":
                door_side = "right"
                door_x = cyl_end_x
            elif sealed_end == "right":
                door_side = "left"
                door_x = cyl_start_x
            else:
                door_side = "right"
                door_x = cyl_end_x
            door_y = cyl_start_y + (cyl_width // 2)

            # Draw the cylinder's border (walls) with an opening for the door.
            for y in range(cyl_start_y, cyl_start_y + cyl_width):
                for x in range(cyl_start_x, cyl_start_x + cyl_length):
                    # Determine if (x, y) is in the border region.
                    if (x < cyl_start_x + self._border_width or 
                        x > cyl_end_x - self._border_width or 
                        y < cyl_start_y + self._border_width or 
                        y > cyl_end_y - self._border_width):
                        # Leave a door opening on the designated door side.
                        if (x, y) == (door_x, door_y):
                            self._grid[y, x] = "door"
                        else:
                            self._grid[y, x] = self._border_object
                            self._wall_positions.add((x, y))
            # Collect interior cells (those not part of the border).
            interior = set()
            for y in range(cyl_start_y + self._border_width, cyl_start_y + cyl_width - self._border_width):
                for x in range(cyl_start_x + self._border_width, cyl_start_x + cyl_length - self._border_width):
                    if self._grid[y, x] == "empty":
                        interior.add((x, y))
            # Place the special resource in a random interior cell.
            if special_item and interior:
                chosen = self._rng.choice(list(interior))
                self._grid[chosen[1], chosen[0]] = special_item

            # For the first cylinder, place the agent outside the door.
            if i == 0:
                if door_side == "right":
                    agent_x = door_x + 1
                    agent_y = door_y
                else:
                    agent_x = door_x - 1
                    agent_y = door_y
                if 0 <= agent_x < self._overall_width and 0 <= agent_y < self._overall_height:
                    self._grid[agent_y, agent_x] = "agent.agent"
                else:
                    # Fallback: choose any empty cell.
                    empty_cells = [
                        (x, y)
                        for y in range(self._overall_height)
                        for x in range(self._overall_width)
                        if self._grid[y, x] == "empty"
                    ]
                    if empty_cells:
                        pos = self._rng.choice(empty_cells)
                        self._grid[pos[1], pos[0]] = "agent.agent"

        return self._grid
