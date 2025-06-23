"""
MinGrid-Inspired Empty Navigation Environment

Inspired by MinGrid's EmptyEnv. This creates a simple empty room where agents must navigate
to reach altar(s) that provide heart rewards. This is the simplest navigation task, useful
for validating RL algorithms and testing basic movement and goal-reaching behavior.

Original MinGrid EmptyEnv: Agent navigates empty room to reach green goal square.
MettagGrid adaptation: Agent navigates empty room to reach altar(s) for heart rewards.

The environment consists of:
- An empty room with wall borders
- One or more altars placed at strategic locations (typically far from agent spawn)
- Agents spawn at random or fixed positions
- Reward is given when agents reach and use altars
"""

from typing import Optional, Tuple
import numpy as np
from omegaconf import DictConfig

from mettagrid.room.room import Room


class MinigridEmptyNavigation(Room):
    def __init__(
        self,
        width: int,
        height: int,
        objects: DictConfig,
        agents: int | DictConfig = 1,
        seed: Optional[int] = None,
        border_width: int = 1,
        border_object: str = "wall",
        altar_placement: str = "corner",  # "corner", "center", "random", "edges"
        team: str | None = None,
    ):
        super().__init__(border_width=border_width, border_object=border_object, labels=["minigrid_empty_nav"])
        self.set_size_labels(width, height)
        self._rng = np.random.default_rng(seed)
        self._width = width
        self._height = height
        self._agents = agents
        self._objects = objects
        self._altar_placement = altar_placement
        self._team = team

    def _build(self) -> np.ndarray:
        # Create empty grid
        grid = np.full((self._height, self._width), "empty", dtype="<U50")

        # Track occupied positions
        occupied = set()

        # Place altars based on placement strategy
        altar_count = self._objects.get("altar", 1)
        altar_positions = self._get_altar_positions(altar_count)

        for pos in altar_positions:
            r, c = pos
            grid[r, c] = "altar"
            occupied.add(pos)

        # Place agents
        agent_positions = self._get_agent_positions(occupied)
        if isinstance(self._agents, int):
            if self._team is None:
                agents = ["agent.agent"] * self._agents
            else:
                agents = [f"agent.{self._team}"] * self._agents
        elif isinstance(self._agents, DictConfig):
            agents = [f"agent.{agent}" for agent, na in self._agents.items() for _ in range(na)]

        for i, agent in enumerate(agents):
            if i < len(agent_positions):
                r, c = agent_positions[i]
                grid[r, c] = agent
                occupied.add((r, c))

        # Place any additional objects
        for obj_name, obj_count in self._objects.items():
            if obj_name == "altar":
                continue  # Already placed

            current_count = np.sum(grid == obj_name)
            remaining_count = obj_count - current_count

            for _ in range(remaining_count):
                pos = self._get_random_empty_position(occupied)
                if pos is not None:
                    r, c = pos
                    grid[r, c] = obj_name
                    occupied.add(pos)

        return grid

    def _get_altar_positions(self, count: int) -> list[Tuple[int, int]]:
        """Get altar positions based on placement strategy."""
        positions = []

        if self._altar_placement == "corner":
            # Place in corners, starting from bottom-right (like MinGrid)
            corners = [
                (self._height - 2, self._width - 2),  # bottom-right
                (1, 1),                               # top-left
                (1, self._width - 2),                 # top-right
                (self._height - 2, 1),                # bottom-left
            ]
            for i in range(min(count, len(corners))):
                positions.append(corners[i])

        elif self._altar_placement == "center":
            # Place in center area
            center_r, center_c = self._height // 2, self._width // 2
            positions.append((center_r, center_c))

            # If more altars needed, place around center
            offsets = [(0, 2), (0, -2), (2, 0), (-2, 0), (2, 2), (-2, -2), (2, -2), (-2, 2)]
            for i in range(1, count):
                if i - 1 < len(offsets):
                    dr, dc = offsets[i - 1]
                    r, c = center_r + dr, center_c + dc
                    if 1 <= r < self._height - 1 and 1 <= c < self._width - 1:
                        positions.append((r, c))

        elif self._altar_placement == "edges":
            # Place along edges but not corners
            edge_positions = []
            # Top and bottom edges
            for c in range(2, self._width - 2):
                edge_positions.extend([(1, c), (self._height - 2, c)])
            # Left and right edges
            for r in range(2, self._height - 2):
                edge_positions.extend([(r, 1), (r, self._width - 2)])

            selected = self._rng.choice(len(edge_positions), size=min(count, len(edge_positions)), replace=False)
            positions = [edge_positions[i] for i in selected]

        else:  # random
            for _ in range(count):
                r = self._rng.integers(1, self._height - 1)
                c = self._rng.integers(1, self._width - 1)
                positions.append((r, c))

        return positions

    def _get_agent_positions(self, occupied: set) -> list[Tuple[int, int]]:
        """Get agent spawn positions, typically opposite from altars."""
        positions = []

        if self._altar_placement == "corner":
            # If altar is in bottom-right, spawn agents in top-left area
            spawn_area = [(r, c) for r in range(1, 4) for c in range(1, 4)
                         if (r, c) not in occupied]
        else:
            # Random spawning away from occupied positions
            spawn_area = [(r, c) for r in range(1, self._height - 1)
                         for c in range(1, self._width - 1)
                         if (r, c) not in occupied]

        if spawn_area:
            selected_indices = self._rng.choice(len(spawn_area),
                                             size=min(self._agents, len(spawn_area)),
                                             replace=False)
            positions = [spawn_area[i] for i in selected_indices]

        return positions

    def _get_random_empty_position(self, occupied: set) -> Optional[Tuple[int, int]]:
        """Get a random empty position not in occupied set."""
        empty_positions = [(r, c) for r in range(1, self._height - 1)
                          for c in range(1, self._width - 1)
                          if (r, c) not in occupied]

        if empty_positions:
            return tuple(self._rng.choice(empty_positions))
        return None
