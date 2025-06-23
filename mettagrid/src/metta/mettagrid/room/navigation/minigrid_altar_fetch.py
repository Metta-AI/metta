"""
MinGrid-Inspired Altar Fetch Environment

Inspired by MinGrid's FetchEnv. This creates an environment with multiple altars of different
types scattered around. Agents must collect hearts from specific altars or collect a certain
number of hearts total.

Original MinGrid FetchEnv: Multiple colored objects, agent must pick up specific one based on text mission.
MettagGrid adaptation: Multiple altars, agent must collect hearts from them (can be specific types or total count).

The environment consists of:
- An empty room with wall borders
- Multiple altars scattered around the room
- Agents spawn at random positions
- Reward is given when agents collect hearts from altars
- Can have different altar types (using different colors or positions)
"""

from typing import Optional, Tuple, List
import numpy as np
from omegaconf import DictConfig

from mettagrid.room.room import Room


class MinigridAltarFetch(Room):
    def __init__(
        self,
        width: int,
        height: int,
        objects: DictConfig,
        agents: int | DictConfig = 1,
        seed: Optional[int] = None,
        border_width: int = 1,
        border_object: str = "wall",
        altar_placement: str = "scattered",  # "scattered", "grid", "clustered"
        min_altar_distance: int = 3,  # Minimum distance between altars
        team: str | None = None,
    ):
        super().__init__(border_width=border_width, border_object=border_object, labels=["minigrid_altar_fetch"])
        self.set_size_labels(width, height)
        self._rng = np.random.default_rng(seed)
        self._width = width
        self._height = height
        self._agents = agents
        self._objects = objects
        self._altar_placement = altar_placement
        self._min_altar_distance = min_altar_distance
        self._team = team

    def _build(self) -> np.ndarray:
        # Create empty grid
        grid = np.full((self._height, self._width), "empty", dtype="<U50")

        # Track occupied positions
        occupied = set()

        # Place altars with proper spacing
        altar_count = self._objects.get("altar", 3)
        altar_positions = self._get_altar_positions(altar_count)

        for pos in altar_positions:
            r, c = pos
            grid[r, c] = "altar"
            occupied.add(pos)

        # Place generators if specified (additional collectible sources)
        generator_count = self._objects.get("generator", 0)
        if generator_count > 0:
            generator_positions = self._get_generator_positions(generator_count, occupied)
            for pos in generator_positions:
                r, c = pos
                grid[r, c] = "generator"
                occupied.add(pos)

        # Place agents away from altars
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
            if obj_name in ["altar", "generator"]:
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

    def _get_altar_positions(self, count: int) -> List[Tuple[int, int]]:
        """Get altar positions based on placement strategy with proper spacing."""
        positions = []

        if self._altar_placement == "grid":
            # Place altars in a grid pattern
            rows = int(np.sqrt(count))
            cols = (count + rows - 1) // rows

            row_spacing = max(self._min_altar_distance, (self._height - 2) // (rows + 1))
            col_spacing = max(self._min_altar_distance, (self._width - 2) // (cols + 1))

            for i in range(count):
                row_idx = i // cols
                col_idx = i % cols
                r = 1 + (row_idx + 1) * row_spacing
                c = 1 + (col_idx + 1) * col_spacing

                # Clamp to valid bounds
                r = min(r, self._height - 2)
                c = min(c, self._width - 2)
                positions.append((r, c))

        elif self._altar_placement == "clustered":
            # Place altars in clusters
            num_clusters = max(1, count // 3)
            altars_per_cluster = count // num_clusters
            extra_altars = count % num_clusters

            for cluster in range(num_clusters):
                # Choose cluster center
                center_r = self._rng.integers(3, self._height - 3)
                center_c = self._rng.integers(3, self._width - 3)

                altars_in_this_cluster = altars_per_cluster + (1 if cluster < extra_altars else 0)

                # Place altars around cluster center
                for _ in range(altars_in_this_cluster):
                    # Find position near cluster center
                    attempts = 0
                    while attempts < 20:
                        dr = self._rng.integers(-2, 3)
                        dc = self._rng.integers(-2, 3)
                        r, c = center_r + dr, center_c + dc

                        if (1 <= r < self._height - 1 and 1 <= c < self._width - 1 and
                            self._is_position_valid(r, c, positions)):
                            positions.append((r, c))
                            break
                        attempts += 1

        else:  # scattered (default)
            # Place altars randomly with minimum distance constraint
            for _ in range(count):
                attempts = 0
                while attempts < 50:  # Limit attempts to avoid infinite loop
                    r = self._rng.integers(1, self._height - 1)
                    c = self._rng.integers(1, self._width - 1)

                    if self._is_position_valid(r, c, positions):
                        positions.append((r, c))
                        break
                    attempts += 1

        return positions

    def _get_generator_positions(self, count: int, occupied: set) -> List[Tuple[int, int]]:
        """Place generators randomly, avoiding occupied positions."""
        positions = []

        for _ in range(count):
            pos = self._get_random_empty_position(occupied)
            if pos is not None:
                positions.append(pos)
                occupied.add(pos)

        return positions

    def _is_position_valid(self, r: int, c: int, existing_positions: List[Tuple[int, int]]) -> bool:
        """Check if position is valid (minimum distance from existing positions)."""
        for er, ec in existing_positions:
            distance = abs(r - er) + abs(c - ec)  # Manhattan distance
            if distance < self._min_altar_distance:
                return False
        return True

    def _get_agent_positions(self, occupied: set) -> List[Tuple[int, int]]:
        """Get agent spawn positions, avoiding occupied areas."""
        positions = []

        # Try to place agents away from altars/generators
        for i in range(self._agents):
            attempts = 0
            found_position = False

            while attempts < 30:
                r = self._rng.integers(1, self._height - 1)
                c = self._rng.integers(1, self._width - 1)

                if (r, c) not in occupied:
                    # Check if reasonably far from occupied positions
                    min_distance = min([abs(r - or_) + abs(c - oc) for or_, oc in occupied], default=float('inf'))
                    if min_distance >= 2:  # At least 2 cells away
                        positions.append((r, c))
                        occupied.add((r, c))
                        found_position = True
                        break
                attempts += 1

            # If couldn't find good position, just place anywhere empty
            if not found_position:
                pos = self._get_random_empty_position(occupied)
                if pos is not None:
                    positions.append(pos)
                    occupied.add(pos)

        return positions

    def _get_random_empty_position(self, occupied: set) -> Optional[Tuple[int, int]]:
        """Get a random empty position not in occupied set."""
        empty_positions = [(r, c) for r in range(1, self._height - 1)
                          for c in range(1, self._width - 1)
                          if (r, c) not in occupied]

        if empty_positions:
            return tuple(self._rng.choice(empty_positions))
        return None
