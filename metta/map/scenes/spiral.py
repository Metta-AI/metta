import numpy as np

from metta.common.util.config import Config
from metta.map.scene import Scene


class SpiralParams(Config):
    objects: dict[str, int] = {}
    agents: int | dict[str, int] = 0
    spacing: int = 15  # Minimum spacing between objects along the spiral
    start_radius: int = 0  # Starting radius from center
    radius_increment: float = 2.5  # How much the radius increases per turn
    angle_increment: float = 0.3  # Angle increment in radians
    randomize_position: int = 2  # Random offset in cells for each position
    place_at_center: bool = True  # Whether to place first object at exact center


class Spiral(Scene[SpiralParams]):
    """
    Place objects along a spiral path emanating from the center of the map.

    This scene ensures proper spacing between objects so that agents with
    limited view distance can only see one object at a time when following
    the spiral path.
    """

    def render(self):
        height, width, params = self.height, self.width, self.params

        # Center of the map
        cx, cy = width // 2, height // 2

        # Collect all objects to place
        symbols = []
        for obj_name, count in params.objects.items():
            symbols.extend([obj_name] * count)

        # Add agents
        if isinstance(params.agents, int):
            agents = ["agent.agent"] * params.agents
        elif isinstance(params.agents, dict):
            agents = ["agent." + str(agent) for agent, na in params.agents.items() for _ in range(na)]
        else:
            raise ValueError(f"Invalid agents: {params.agents}")

        # Determine placement order - agents first if placing at center
        if params.place_at_center and agents:
            all_symbols = agents + symbols
        else:
            all_symbols = symbols + agents

        if not all_symbols:
            return

        # Generate spiral positions
        positions = []
        angle = 0
        radius = params.start_radius

        # Place first item at center if requested
        if params.place_at_center and all_symbols:
            positions.append((cx, cy))
            all_symbols = all_symbols[1:]
            angle += params.angle_increment

        # Generate remaining positions along spiral
        last_x, last_y = cx, cy
        while len(positions) < len(all_symbols) + 1:  # +1 for the center position already added
            # Calculate next position on spiral
            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle))

            # Check if we've moved far enough from the last position
            distance = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)

            if distance >= params.spacing:
                # Add randomization if requested
                if params.randomize_position > 0:
                    offset_x = self.rng.integers(-params.randomize_position, params.randomize_position + 1)
                    offset_y = self.rng.integers(-params.randomize_position, params.randomize_position + 1)
                    x = np.clip(x + offset_x, 1, width - 2)
                    y = np.clip(y + offset_y, 1, height - 2)

                # Check bounds
                if 0 <= x < width and 0 <= y < height:
                    positions.append((x, y))
                    last_x, last_y = x, y

            # Update spiral parameters
            angle += params.angle_increment
            radius += params.radius_increment * params.angle_increment / (2 * np.pi)

            # Safety check to prevent infinite loop
            if radius > max(width, height):
                break

        # Place symbols at the calculated positions
        for i, symbol in enumerate(all_symbols):
            if i < len(positions):
                x, y = positions[i]
                # Only place if the cell is empty
                if self.grid[y, x] == "empty":
                    self.grid[y, x] = symbol
