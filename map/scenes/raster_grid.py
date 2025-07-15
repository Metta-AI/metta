from typing import List, Tuple

from metta.config import Config
from metta.map.scene import Scene


class RasterGridParams(Config):
    """Parameters for the RasterGrid scene."""

    objects: dict[str, int] = {"altar": 25}
    agents: int | dict[str, int] = 0
    grid_spacing: int = 6
    wall_margin: int = 1


class RasterGrid(Scene[RasterGridParams]):
    """Scene that places altars at regular grid intersections to teach rastering behavior.

    Altars are placed at the intersections of a regular grid pattern across the map.
    This encourages agents to learn efficient grid-based search patterns.
    """

    def _get_intersection_points(self) -> List[Tuple[int, int]]:
        """Calculate all valid grid intersection points."""
        points = []

        # Calculate grid lines (starting from wall_margin to avoid walls)
        x_start = self.params.wall_margin
        y_start = self.params.wall_margin
        x_end = self.width - self.params.wall_margin
        y_end = self.height - self.params.wall_margin

        # Generate all intersection points
        for x in range(x_start, x_end, self.params.grid_spacing):
            for y in range(y_start, y_end, self.params.grid_spacing):
                points.append((x, y))

        return points

    def render(self):
        """Render the scene with objects at grid intersections."""
        params = self.params

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

        # Place agents first, then objects
        all_symbols = agents + symbols

        if not all_symbols:
            return

        # Get all possible intersection points
        intersection_points = self._get_intersection_points()

        # Randomly select positions from the intersection points
        if len(intersection_points) < len(all_symbols):
            print(
                f"Warning: Only {len(intersection_points)} intersection points available, "
                f"but {len(all_symbols)} items requested. Using all available points."
            )
            selected_indices = list(range(len(intersection_points)))
        else:
            selected_indices = self.rng.choice(len(intersection_points), size=len(all_symbols), replace=False)

        # Place symbols at the selected intersection points
        for i, symbol in enumerate(all_symbols):
            if i < len(selected_indices):
                point_idx = selected_indices[i]
                x, y = intersection_points[point_idx]
                # Only place if the cell is empty
                if self.grid[y, x] == "empty":
                    self.grid[y, x] = symbol
