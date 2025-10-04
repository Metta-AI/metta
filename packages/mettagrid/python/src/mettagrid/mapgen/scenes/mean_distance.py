import numpy as np

from mettagrid.mapgen.scene import Scene, SceneConfig


class MeanDistanceConfig(SceneConfig):
    mean_distance: float  # Mean distance parameter for objects relative to agent.
    objects: dict[str, int]


class MeanDistance(Scene[MeanDistanceConfig]):
    """
    This scene places an agent at the center of the scene and places objects at a mean distance from the agent.
    """

    def render(self):
        # Define the agent's initial position (here: center of the room)
        agent_pos = (self.height // 2, self.width // 2)

        # Place the agent at the center.
        self.grid[agent_pos] = "agent.agent"

        # Place each object based on a Poisson-distributed distance from the agent.
        # For each object type and the number of instances required:
        for obj_name, count in self.config.objects.items():
            placed = 0
            while placed < count:
                # Sample a distance from a Poisson distribution.
                d = self.rng.poisson(lam=self.config.mean_distance)
                # Ensure a nonzero distance (so objects don't collide with the agent)
                if d == 0:
                    d = 1
                # Sample an angle uniformly from 0 to 2*pi.
                angle = self.rng.uniform(0, 2 * np.pi)
                # Convert polar coordinates to grid offsets.
                dx = int(round(d * np.cos(angle)))
                dy = int(round(d * np.sin(angle)))
                # Candidate position (note: grid indexing is row, col so we add dy then dx).
                candidate = (agent_pos[0] + dy, agent_pos[1] + dx)
                # Check if candidate position is inside the room and unoccupied.
                if (
                    0 <= candidate[0] < self.height
                    and 0 <= candidate[1] < self.width
                    and self.grid[candidate] == "empty"
                ):
                    self.grid[candidate] = obj_name
                    placed += 1
