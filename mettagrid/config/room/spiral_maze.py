from math import sqrt
import numpy as np
from omegaconf import DictConfig
from mettagrid.config.room.room import Room

class SpiralMaze(Room):
    def __init__(
        self,
        width: int,
        height: int,
        border_width: int = 1,
        corridor_width: int = 2,
        resource_density: float = 0.5,
        seed = None,
        agents: int = 1,
        **kwargs,
    ):
        """
        Creates a spiral maze environment with walls.
        
        The maze is built on a grid initially filled with walls.
        A spiral corridor is carved starting at the center and winding outward.
        Within this corridor, resources (heart altars, convertors, and generators) are scattered,
        with a higher density near the center.
        The agent is placed at (or near) the center.
        """
        super().__init__(border_width=border_width, border_object="wall")
        self.width = width
        self.height = height
        self.border_width = border_width
        self.corridor_width = corridor_width
        self.resource_density = resource_density
        self.seed = seed
        self.agents = agents if isinstance(agents, int) else agents
        self._rng = np.random.default_rng(seed)
        # Create a grid completely filled with walls.
        self._grid = np.full((self.height, self.width), "wall", dtype='<U50')

    def _build(self) -> np.ndarray:
        grid = self._grid.copy()
        cx, cy = self.width // 2, self.height // 2  # center of the grid

        # Compute maximum distance from the center (for normalizing resource probability)
        max_dx = max(cx, self.width - 1 - cx)
        max_dy = max(cy, self.height - 1 - cy)
        max_distance = sqrt(max_dx**2 + max_dy**2)

        # Generate spiral coordinates starting at the center, moving outward.
        def generate_spiral(center_x, center_y, width, height):
            x, y = center_x, center_y
            dx, dy = 1, 0  # start moving right
            steps = 1
            coords = [(x, y)]
            while True:
                for _ in range(2):  # two legs for each step count
                    for _ in range(steps):
                        x += dx
                        y += dy
                        if not (0 <= x < width and 0 <= y < height):
                            return coords
                        coords.append((x, y))
                    # Rotate direction 90° clockwise: (dx, dy) -> (dy, -dx)
                    dx, dy = dy, -dx
                steps += 1
            return coords

        spiral_coords = generate_spiral(cx, cy, self.width, self.height)

        # Carve the spiral corridor from the center outwards.
        # For each coordinate along the spiral, carve a block of size (corridor_width x corridor_width)
        for (x, y) in spiral_coords:
            for dx in range(self.corridor_width):
                for dy in range(self.corridor_width):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        grid[ny, nx] = "empty"

        # Scatter resources in the carved corridor.
        # The probability is higher near the center.
        # We use weighted probabilities: heart altar: 40%, convertor: 40%, generator: 20%.
        resources = ["generator", "heart altar", "convertor"]
        resource_weights = [0.2, 0.4, 0.4]
        for y in range(self.height):
            for x in range(self.width):
                if grid[y, x] == "empty":
                    distance = sqrt((x - cx)**2 + (y - cy)**2)
                    norm = min(distance / max_distance, 1.0)
                    # Probability decays linearly with distance from the center.
                    prob = self.resource_density * (1 - norm)
                    if self._rng.random() < prob:
                        grid[y, x] = self._rng.choice(resources, p=resource_weights)

        # Place the agent at (or near) the center.
        if grid[cy, cx] == "empty":
            grid[cy, cx] = "agent.agent"
        else:
            # If the center isn't empty, search outward for an empty cell.
            for offset in range(1, max(cx, cy) + 1):
                for dx in range(-offset, offset + 1):
                    for dy in range(-offset, offset + 1):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height and grid[ny, nx] == "empty":
                            grid[ny, nx] = "agent.agent"
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break

        return grid
