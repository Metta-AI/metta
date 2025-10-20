import numpy as np

from mettagrid.mapgen.scene import Scene, SceneConfig


class SpiralConfig(SceneConfig):
    objects: dict[str, int] = {}
    agents: int | dict[str, int] = 0
    spacing: int = 15  # Minimum spacing between objects along the spiral
    start_radius: int = 0  # Starting radius from center
    radius_increment: float = 2.5  # How much the radius increases per turn
    angle_increment: float = 0.3  # Angle increment in radians
    randomize_position: int = 2  # Random offset in cells for each position
    place_at_center: bool = True  # Whether to place first object at exact center


class Spiral(Scene[SpiralConfig]):
    """
    Place objects along a spiral path emanating from the center of the map.

    This scene ensures proper spacing between objects so that agents with
    limited view distance can only see one object at a time when following
    the spiral path.
    """

    def render(self):
        height, width, config = self.height, self.width, self.config

        # Center of the map
        cx, cy = width // 2, height // 2

        # Collect all objects to place
        symbols = []
        for obj_name, count in config.objects.items():
            symbols.extend([obj_name] * count)

        # Add agents
        if isinstance(config.agents, int):
            agents = ["agent.agent"] * config.agents
        elif isinstance(config.agents, dict):
            agents = ["agent." + str(agent) for agent, na in config.agents.items() for _ in range(na)]
        else:
            raise ValueError(f"Invalid agents: {config.agents}")

        # Determine placement order - agents first if placing at center
        if config.place_at_center and agents:
            all_symbols = agents + symbols
        else:
            all_symbols = symbols + agents

        if not all_symbols:
            return

        # Generate spiral positions
        positions = []
        angle = 0
        radius = config.start_radius

        # Place first item at center if requested
        if config.place_at_center and all_symbols:
            positions.append((cx, cy))
            # Don't remove the first symbol - we still need to place it!
            # all_symbols = all_symbols[1:]
            angle += config.angle_increment

        # Generate remaining positions along spiral
        last_x, last_y = cx, cy
        while len(positions) < len(all_symbols):  # Remove the +1 since we're not skipping the first symbol
            # Calculate next position on spiral
            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle))

            # Check if we've moved far enough from the last position
            distance = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)

            if distance >= config.spacing:
                # Add randomization if requested
                if config.randomize_position > 0:
                    offset_x = self.rng.integers(-config.randomize_position, config.randomize_position + 1)
                    offset_y = self.rng.integers(-config.randomize_position, config.randomize_position + 1)
                    x = np.clip(x + offset_x, 1, width - 2)
                    y = np.clip(y + offset_y, 1, height - 2)

                # Check bounds
                if 0 <= x < width and 0 <= y < height:
                    positions.append((x, y))
                    last_x, last_y = x, y

            # Update spiral parameters
            angle += config.angle_increment
            radius += config.radius_increment * config.angle_increment / (2 * np.pi)

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
