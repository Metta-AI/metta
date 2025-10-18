"""
Adapter for extracting alignment data from MettaGrid environments.

Provides utilities to extract positions, compute task directions, and
prepare data for GAMMA metric computation.
"""

from typing import Any

import numpy as np
import numpy.typing as npt


class MettaGridAdapter:
    """
    Adapter for extracting alignment-relevant data from MettaGrid.

    Handles the conversion from MettaGrid's grid-based coordinates
    to continuous space for GAMMA metrics.
    """

    def __init__(self, grid_to_continuous_scale: float = 1.0):
        """
        Initialize adapter.

        Args:
            grid_to_continuous_scale: Scale factor for grid→continuous conversion
        """
        self.scale = grid_to_continuous_scale

    def extract_agent_positions(self, env: Any) -> npt.NDArray[np.floating[Any]]:
        """
        Extract agent positions from MettaGrid environment.

        Args:
            env: MettaGrid environment instance

        Returns:
            Agent positions, shape (num_agents, 2) in continuous coordinates
        """
        grid_objects = env.grid_objects()

        # Collect agent positions
        agent_positions = {}
        for _obj_id, obj_data in grid_objects.items():
            if "agent_id" in obj_data:
                agent_id = obj_data["agent_id"]
                # Grid coordinates (row, col)
                r = obj_data["r"]
                c = obj_data["c"]
                # Convert to continuous (x, y) - note: x=col, y=row
                agent_positions[agent_id] = np.array([c, r], dtype=np.float32) * self.scale

        # Sort by agent_id and return as array
        num_agents = max(agent_positions.keys()) + 1 if agent_positions else 0
        positions = np.zeros((num_agents, 2), dtype=np.float32)

        for agent_id, pos in agent_positions.items():
            positions[agent_id] = pos

        return positions

    def compute_task_directions_to_resources(
        self,
        env: Any,
        resource_types: list[str] | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """
        Compute task directions pointing toward nearest resources.

        Args:
            env: MettaGrid environment
            resource_types: List of resource type names to target (e.g., ['generator', 'converter'])

        Returns:
            Task directions, shape (num_agents, 2)
        """
        grid_objects = env.grid_objects()

        # Extract agent positions
        agent_positions = {}
        for _obj_id, obj_data in grid_objects.items():
            if "agent_id" in obj_data:
                agent_id = obj_data["agent_id"]
                r, c = obj_data["r"], obj_data["c"]
                agent_positions[agent_id] = np.array([c, r], dtype=np.float32)

        # Extract resource positions
        resource_positions = []
        for _obj_id, obj_data in grid_objects.items():
            obj_type = obj_data.get("type", "")
            # Check if this is a resource we care about
            if resource_types is None or any(rt in str(obj_type).lower() for rt in resource_types):
                if "r" in obj_data and "c" in obj_data:
                    r, c = obj_data["r"], obj_data["c"]
                    resource_positions.append(np.array([c, r], dtype=np.float32))

        # Compute task directions
        num_agents = max(agent_positions.keys()) + 1 if agent_positions else 0
        task_directions = np.zeros((num_agents, 2), dtype=np.float32)

        if len(resource_positions) == 0:
            # No resources found, return zero directions
            return task_directions

        resource_array = np.array(resource_positions)

        for agent_id, agent_pos in agent_positions.items():
            # Find nearest resource
            distances = np.linalg.norm(resource_array - agent_pos, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_resource = resource_array[nearest_idx]

            # Compute direction
            direction = nearest_resource - agent_pos
            distance = np.linalg.norm(direction)

            if distance > 1e-6:
                task_directions[agent_id] = direction / distance
            else:
                task_directions[agent_id] = np.zeros(2)

        # Scale to continuous space
        task_directions = task_directions * self.scale

        return task_directions

    def compute_task_directions_to_goal(
        self,
        agent_positions: npt.NDArray[np.floating[Any]],
        goals: npt.NDArray[np.floating[Any]],
    ) -> npt.NDArray[np.floating[Any]]:
        """
        Compute task directions pointing toward goals.

        Args:
            agent_positions: Current agent positions, shape (num_agents, 2)
            goals: Goal positions, shape (num_agents, 2)

        Returns:
            Task directions, shape (num_agents, 2)
        """
        directions = goals - agent_positions
        distances = np.linalg.norm(directions, axis=1, keepdims=True)

        # Avoid division by zero
        distances = np.maximum(distances, 1e-8)

        task_directions = directions / distances

        return task_directions

    def compute_task_directions_formation(
        self,
        agent_positions: npt.NDArray[np.floating[Any]],
        desired_formation: npt.NDArray[np.floating[Any]],
    ) -> npt.NDArray[np.floating[Any]]:
        """
        Compute task directions for formation tasks.

        Args:
            agent_positions: Current positions, shape (num_agents, 2)
            desired_formation: Desired formation positions, shape (num_agents, 2)

        Returns:
            Task directions toward formation positions
        """
        return self.compute_task_directions_to_goal(agent_positions, desired_formation)
