import numpy as np

from mettagrid.mapgen.scene import Scene, SceneConfig


class GridAltarsConfig(SceneConfig):
    objects: dict[str, int] = {}
    agents: int | dict[str, int] = 0
    grid_rows: int = 3  # Number of rows in the grid
    grid_cols: int = 3  # Number of columns in the grid
    margin: int = 5  # Margin from map edges
    min_spacing: int = 9  # Minimum spacing between grid points
    randomize_position: int = 0  # Random offset for each position
    place_agent_center: bool = True  # Whether to place agent at center


class GridAltars(Scene[GridAltarsConfig]):
    """
    Place objects on a grid pattern across the map.

    This scene creates a regular grid of positions and places objects
    (typically altars) at grid nodes with optional randomization.
    """

    def render(self):
        height, width, config = self.height, self.width, self.config

        # Validate min_spacing to prevent division by zero
        if config.min_spacing <= 0:
            raise ValueError(f"min_spacing must be greater than 0, got {config.min_spacing}")

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

        if not symbols and not agents:
            return

            # Calculate grid positions
        usable_width = width - 2 * config.margin
        usable_height = height - 2 * config.margin

        # Calculate actual number of rows/cols that fit with minimum spacing
        max_cols = max(1, 1 + usable_width // config.min_spacing)
        max_rows = max(1, 1 + usable_height // config.min_spacing)

        actual_cols = min(config.grid_cols, max_cols)
        actual_rows = min(config.grid_rows, max_rows)

        # Calculate actual spacing to evenly distribute the grid
        if actual_cols > 1:
            x_spacing = usable_width / (actual_cols - 1)
        else:
            x_spacing = 0

        if actual_rows > 1:
            y_spacing = usable_height / (actual_rows - 1)
        else:
            y_spacing = 0

        # Generate grid positions
        positions = []
        for row in range(actual_rows):
            for col in range(actual_cols):
                # Calculate position without randomization first
                if actual_cols > 1:
                    x = config.margin + int(col * x_spacing)
                else:
                    x = width // 2

                if actual_rows > 1:
                    y = config.margin + int(row * y_spacing)
                else:
                    y = height // 2

                # Add randomization only if requested
                if config.randomize_position > 0:
                    offset_x = self.rng.integers(-config.randomize_position, config.randomize_position + 1)
                    offset_y = self.rng.integers(-config.randomize_position, config.randomize_position + 1)
                    x = np.clip(x + offset_x, 1, width - 2)
                    y = np.clip(y + offset_y, 1, height - 2)

                positions.append((x, y))

        # Place agent at center if requested
        if config.place_agent_center and agents:
            cx, cy = width // 2, height // 2
            if self.grid[cy, cx] == "empty":
                self.grid[cy, cx] = agents[0]
                agents = agents[1:]

        # Shuffle positions for random placement
        self.rng.shuffle(positions)

        # Place remaining agents first if not centering
        if not config.place_agent_center:
            for agent in agents:
                if positions:
                    x, y = positions.pop(0)
                    if self.grid[y, x] == "empty":
                        self.grid[y, x] = agent

        # Place objects at remaining positions
        for symbol in symbols:
            if positions:
                x, y = positions.pop(0)
                if self.grid[y, x] == "empty":
                    self.grid[y, x] = symbol
