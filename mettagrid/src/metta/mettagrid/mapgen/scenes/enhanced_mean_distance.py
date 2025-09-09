"""Enhanced MeanDistance scene using int-based storage and type constants.

This demonstrates updating geometric placement scenes to support both
legacy string-based and new int-based grids.
"""

import numpy as np

from metta.mettagrid.config import Config
from metta.mettagrid.mapgen.enhanced_scene import EnhancedScene
from metta.mettagrid.object_types import ObjectTypes


class EnhancedMeanDistanceParams(Config):
    """Parameters for enhanced mean distance scene generation."""

    objects: dict[str, int] = {}
    mean_distance: float = 5.0
    max_placement_attempts: int = 1000


class EnhancedMeanDistance(EnhancedScene[EnhancedMeanDistanceParams]):
    """
    Enhanced mean distance scene that supports both int-based and string-based grids.

    This scene places an agent at the center and distributes objects at distances
    following a Poisson distribution around the mean distance.

    Key improvements over legacy MeanDistance scene:
    - Type-safe object placement using constants
    - Support for both grid formats during migration
    - Configurable placement attempts for better success rates
    - Better handling of placement failures
    """

    def render(self):
        """Render the mean distance scene with centered agent and distributed objects."""
        params = self.params

        # Define the agent's initial position (center of the area)
        agent_pos = (self.height // 2, self.width // 2)

        # Place the agent at the center
        if self._grid_is_int:
            self.grid[agent_pos] = ObjectTypes.AGENT_DEFAULT
        else:
            self.grid[agent_pos] = "agent.agent"

        # Place each object type with Poisson-distributed distances
        for obj_name, count in params.objects.items():
            self._place_object_type(obj_name, count, agent_pos)

    def _place_object_type(self, obj_name: str, count: int, agent_pos: tuple[int, int]):
        """Place objects of a specific type around the agent position.

        Args:
            obj_name: Name of the object type to place
            count: Number of objects to place
            agent_pos: Position of the central agent (row, col)
        """
        placed = 0
        attempts = 0
        max_attempts = self.params.max_placement_attempts

        # Get the value to place based on grid format
        if self._grid_is_int:
            try:
                placement_value = self.get_object_type_id(obj_name)
            except KeyError:
                print(f"Warning: Unknown object {obj_name}, skipping")
                return
        else:
            placement_value = obj_name

        while placed < count and attempts < max_attempts:
            attempts += 1

            # Sample a distance from a Poisson distribution
            distance = self.rng.poisson(lam=self.params.mean_distance)

            # Ensure minimum distance of 1 to avoid collision with agent
            if distance == 0:
                distance = 1

            # Sample an angle uniformly from 0 to 2*pi
            angle = self.rng.uniform(0, 2 * np.pi)

            # Convert polar coordinates to grid offsets
            dx = int(round(distance * np.cos(angle)))
            dy = int(round(distance * np.sin(angle)))

            # Calculate candidate position
            candidate = (agent_pos[0] + dy, agent_pos[1] + dx)

            # Check if candidate position is valid and empty
            if self._is_valid_placement_position(candidate):
                self.grid[candidate] = placement_value
                placed += 1

        if placed < count:
            print(f"Warning: Could only place {placed}/{count} {obj_name} objects after {max_attempts} attempts")

    def _is_valid_placement_position(self, pos: tuple[int, int]) -> bool:
        """Check if a position is valid for object placement.

        Args:
            pos: Position tuple (row, col)

        Returns:
            True if position is valid and empty
        """
        row, col = pos

        # Check bounds
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False

        # Check if position is empty
        return self.is_empty(row, col)

    def render_with_custom_agent(self, agent_type: str = "agent.agent"):
        """
        Render the scene with a custom agent type.

        Args:
            agent_type: The type of agent to place at center
        """
        params = self.params

        # Define the agent's initial position (center of the area)
        agent_pos = (self.height // 2, self.width // 2)

        # Place the specified agent type at center
        self.set_object(agent_pos[0], agent_pos[1], agent_type)

        # Place objects around the agent
        for obj_name, count in params.objects.items():
            self._place_object_type(obj_name, count, agent_pos)

    def render_with_multiple_agents(self, agent_positions: list[tuple[int, int]], agent_types: list[str] | None = None):
        """
        Render the scene with multiple agents at specified positions.

        Args:
            agent_positions: List of (row, col) positions for agents
            agent_types: Optional list of agent types. If None, uses default agent type.
        """
        params = self.params

        if agent_types is None:
            agent_types = ["agent.agent"] * len(agent_positions)
        elif len(agent_types) != len(agent_positions):
            raise ValueError("agent_types length must match agent_positions length")

        # Place agents
        for pos, agent_type in zip(agent_positions, agent_types, strict=False):
            if self._is_valid_placement_position(pos):
                self.set_object(pos[0], pos[1], agent_type)

        # Place objects around all agents
        all_positions = agent_positions
        for obj_name, count in params.objects.items():
            self._place_objects_around_multiple_centers(obj_name, count, all_positions)

    def _place_objects_around_multiple_centers(
        self, obj_name: str, count: int, center_positions: list[tuple[int, int]]
    ):
        """Place objects around multiple center positions.

        Args:
            obj_name: Name of object type to place
            count: Number of objects to place
            center_positions: List of center positions to place objects around
        """
        placed = 0
        attempts = 0
        max_attempts = self.params.max_placement_attempts

        # Get placement value
        if self._grid_is_int:
            try:
                placement_value = self.get_object_type_id(obj_name)
            except KeyError:
                print(f"Warning: Unknown object {obj_name}, skipping")
                return
        else:
            placement_value = obj_name

        while placed < count and attempts < max_attempts:
            attempts += 1

            # Choose a random center position
            center_pos = self.rng.choice(center_positions)

            # Sample distance and angle
            distance = self.rng.poisson(lam=self.params.mean_distance)
            if distance == 0:
                distance = 1

            angle = self.rng.uniform(0, 2 * np.pi)

            # Convert to grid coordinates
            dx = int(round(distance * np.cos(angle)))
            dy = int(round(distance * np.sin(angle)))
            candidate = (center_pos[0] + dy, center_pos[1] + dx)

            # Place if valid
            if self._is_valid_placement_position(candidate):
                self.grid[candidate] = placement_value
                placed += 1

        if placed < count:
            print(f"Warning: Could only place {placed}/{count} {obj_name} objects after {max_attempts} attempts")

    def get_distance_stats(self) -> dict:
        """Get statistics about object distances from center.

        Returns:
            Dictionary with distance statistics and placement info
        """
        center_pos = (self.height // 2, self.width // 2)

        # Find all non-empty, non-agent positions
        object_positions = []
        object_types = []

        for row in range(self.height):
            for col in range(self.width):
                if not self.is_empty(row, col):
                    # Skip if this is an agent position
                    if self._grid_is_int:
                        if ObjectTypes.is_agent(int(self.grid[row, col])):
                            continue
                        object_types.append(self.get_object_name(int(self.grid[row, col])))
                    else:
                        if str(self.grid[row, col]).startswith("agent."):
                            continue
                        object_types.append(str(self.grid[row, col]))

                    object_positions.append((row, col))

        if not object_positions:
            return {"message": "No objects found in grid"}

        # Calculate distances from center
        distances = []
        for pos in object_positions:
            dx = pos[1] - center_pos[1]  # col difference
            dy = pos[0] - center_pos[0]  # row difference
            distance = np.sqrt(dx * dx + dy * dy)
            distances.append(distance)

        distances = np.array(distances)

        stats = {
            "total_objects": len(object_positions),
            "mean_distance": np.mean(distances),
            "std_distance": np.std(distances),
            "min_distance": np.min(distances),
            "max_distance": np.max(distances),
            "target_mean_distance": self.params.mean_distance,
            "center_position": center_pos,
            "grid_format": "int" if self._grid_is_int else "string",
        }

        # Object type breakdown
        from collections import Counter

        type_counts = Counter(object_types)
        stats["object_type_counts"] = dict(type_counts)

        return stats
