import logging
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.room.room import Room


class MeanDistance(Room):
    def __init__(
        self,
        width: int,
        height: int,
        objects: DictConfig,
        agents: int | DictConfig = 1,
        mean_distance: float = 5.0,  # Mean distance parameter for objects relative to agent.
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: str = "wall",
    ):
        super().__init__(border_width=border_width, border_object=border_object, labels=["mean_distance"])
        logger = logging.getLogger(__name__)

        self._rng = np.random.default_rng(seed)
        self._width = width
        self._height = height
        self._objects = objects
        self._agents = agents
        self.set_size_labels(width, height)
        if mean_distance > width or mean_distance > height:
            logger.warning(
                f"Mean distance {mean_distance} is greater than room size {width}x{height}. "
                f"Setting to {min(width, height) - 1}"
            )
            mean_distance = min(width, height) - 1

        self.mean_distance = mean_distance

    def _build(self):
        # Create an empty room filled with "empty" symbols.
        grid = np.full((self._height, self._width), "empty", dtype=object)

        # Define the agent's initial position (here: center of the room)
        agent_pos = (self._height // 2, self._width // 2)
        # Place the first agent at the center.
        grid[agent_pos] = "agent.agent"

        # Place each object based on a Poisson-distributed distance from the agent.
        # For each object type and the number of instances required:
        for obj_name, count in self._objects.items():
            placed = 0
            while placed < count:
                # Sample a distance from a Poisson distribution.
                d = self._rng.poisson(lam=self.mean_distance)
                # Ensure a nonzero distance (so objects don't collide with the agent)
                if d == 0:
                    d = 1
                # Sample an angle uniformly from 0 to 2*pi.
                angle = self._rng.uniform(0, 2 * np.pi)
                # Convert polar coordinates to grid offsets.
                dx = int(round(d * np.cos(angle)))
                dy = int(round(d * np.sin(angle)))
                # Candidate position (note: grid indexing is row, col so we add dy then dx).
                candidate = (agent_pos[0] + dy, agent_pos[1] + dx)
                # Check if candidate position is inside the room and unoccupied.
                if 0 <= candidate[0] < self._height and 0 <= candidate[1] < self._width and grid[candidate] == "empty":
                    grid[candidate] = obj_name
                    placed += 1

        # If more than one agent is required, place the remaining agents randomly.
        if isinstance(self._agents, int):
            # We already placed one agent at the center.
            for _ in range(1, self._agents):
                while True:
                    free_positions = list(zip(*np.where(grid == "empty"), strict=False))
                    if not free_positions:
                        break
                    pos = free_positions[self._rng.integers(0, len(free_positions))]
                    grid[pos] = "agent.agent"
                    break
        elif isinstance(self._agents, DictConfig):
            # If agents are specified via a DictConfig (e.g. different types),
            # first use the center for the first agent and then assign the rest randomly.
            agent_symbols = []
            for agent, num in self._agents.items():
                agent_symbols.extend([f"agent.{agent}"] * num)
            if agent_symbols:
                grid[agent_pos] = agent_symbols.pop(0)
            while agent_symbols:
                free_positions = list(zip(*np.where(grid == "empty"), strict=False))
                if not free_positions:
                    break
                pos = free_positions[self._rng.integers(0, len(free_positions))]
                grid[pos] = agent_symbols.pop(0)

        return grid
