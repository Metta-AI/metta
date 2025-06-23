"""
MinGrid-Inspired Memory Collection Environment

Inspired by MinGrid's MemoryEnv. This creates a memory test where the agent starts in a room
with an altar (or generator), then navigates through a corridor to reach a split where they
must choose the correct matching object.

Original MinGrid MemoryEnv: Agent sees object, goes through hallway, chooses matching object at split.
MettagGrid adaptation: Agent sees altar/generator type, navigates corridor, chooses matching altar/generator.

The environment consists of:
- Starting room with a "target" object (altar or generator)
- Long corridor connecting to end area
- Split at end with two different objects, one matching the start
- Agent must remember and choose correctly
"""

from typing import Optional, Tuple
import numpy as np
from omegaconf import DictConfig

from mettagrid.room.room import Room


class MinigridMemoryCollection(Room):
    def __init__(
        self,
        width: int,
        height: int,
        objects: DictConfig,
        agents: int | DictConfig = 1,
        seed: Optional[int] = None,
        border_width: int = 1,
        border_object: str = "wall",
        corridor_length: Optional[int] = None,  # Auto-calculated if None
        random_corridor_length: bool = False,
        team: str | None = None,
    ):
        super().__init__(border_width=border_width, border_object=border_object, labels=["minigrid_memory"])
        self.set_size_labels(width, height)
        self._rng = np.random.default_rng(seed)
        self._width = width
        self._height = height
        self._agents = agents
        self._objects = objects
        self._corridor_length = corridor_length
        self._random_corridor_length = random_corridor_length
        self._team = team

    def _build(self) -> np.ndarray:
        # Create empty grid
        grid = np.full((self._height, self._width), "empty", dtype="<U50")

        # Fill with walls initially (we'll carve out rooms and corridors)
        self._create_basic_layout(grid)

        # Place objects in specific locations
        self._place_memory_objects(grid)

        # Place agents in starting room
        self._place_agents(grid)

        return grid

    def _create_basic_layout(self, grid: np.ndarray) -> None:
        """Create the basic layout: start room, corridor, end area with split."""
        # Ensure odd height for proper corridor placement
        assert self._height % 2 == 1, "Height must be odd for proper corridor layout"

        middle_row = self._height // 2

        # Calculate corridor length
        if self._corridor_length is None:
            corridor_length = self._width - 6  # Leave room for start area and end split
        else:
            corridor_length = self._corridor_length

        if self._random_corridor_length:
            corridor_length = self._rng.integers(4, corridor_length + 1)

        # Start room boundaries (left side)
        start_room_width = 4
        upper_room_wall = middle_row - 2
        lower_room_wall = middle_row + 2

        # Create start room walls
        for i in range(1, start_room_width):
            grid[upper_room_wall, i] = "wall"
            grid[lower_room_wall, i] = "wall"
        grid[upper_room_wall + 1, start_room_width - 1] = "wall"
        grid[lower_room_wall - 1, start_room_width - 1] = "wall"

        # Create horizontal corridor walls
        corridor_start = start_room_width
        corridor_end = corridor_start + corridor_length

        for i in range(corridor_start, corridor_end):
            grid[upper_room_wall + 1, i] = "wall"
            grid[lower_room_wall - 1, i] = "wall"

        # Create vertical end wall with gap in middle
        for j in range(self._height):
            if j != middle_row:  # Leave gap in middle for passage
                grid[j, corridor_end] = "wall"

        # Create end split walls
        if corridor_end + 2 < self._width:
            for j in range(self._height):
                grid[j, corridor_end + 2] = "wall"

    def _place_memory_objects(self, grid: np.ndarray) -> None:
        """Place the memory test objects: target in start room, choices at end."""
        middle_row = self._height // 2

        # Choose target object type (altar or generator)
        target_types = ["altar", "generator"]
        target_type = self._rng.choice(target_types)

        # Place target object in start room
        grid[middle_row - 1, 1] = target_type

        # Calculate end positions
        corridor_length = self._width - 6
        corridor_end = 4 + corridor_length

        # Place choice objects at end split
        choice_types = [target_type, target_types[1 - target_types.index(target_type)]]
        self._rng.shuffle(choice_types)  # Randomize which side has correct answer

        # Top choice
        if corridor_end + 1 < self._width and middle_row - 2 >= 0:
            grid[middle_row - 2, corridor_end + 1] = choice_types[0]

        # Bottom choice
        if corridor_end + 1 < self._width and middle_row + 2 < self._height:
            grid[middle_row + 2, corridor_end + 1] = choice_types[1]

        # Store success/failure positions for potential reward logic
        # (This would be used by the environment to track correct choices)
        self._target_type = target_type
        self._success_pos = None
        self._failure_pos = None

        if choice_types[0] == target_type:
            self._success_pos = (middle_row - 1, corridor_end + 1)  # Position in front of correct choice
            self._failure_pos = (middle_row + 1, corridor_end + 1)  # Position in front of wrong choice
        else:
            self._success_pos = (middle_row + 1, corridor_end + 1)  # Position in front of correct choice
            self._failure_pos = (middle_row - 1, corridor_end + 1)  # Position in front of wrong choice

    def _place_agents(self, grid: np.ndarray) -> None:
        """Place agents in the starting room."""
        middle_row = self._height // 2

        # Agent starting positions in start room
        start_positions = [
            (middle_row, 2),  # Center of start room
            (middle_row - 1, 2),  # Slightly above center
            (middle_row + 1, 2),  # Slightly below center
        ]

        if isinstance(self._agents, int):
            if self._team is None:
                agents = ["agent.agent"] * self._agents
            else:
                agents = [f"agent.{self._team}"] * self._agents
        elif isinstance(self._agents, DictConfig):
            agents = [f"agent.{agent}" for agent, na in self._agents.items() for _ in range(na)]

        for i, agent in enumerate(agents):
            if i < len(start_positions):
                r, c = start_positions[i]
                if 0 <= r < self._height and 0 <= c < self._width and grid[r, c] == "empty":
                    grid[r, c] = agent

        # Place any additional objects specified in config
        occupied = set()
        for r in range(self._height):
            for c in range(self._width):
                if grid[r, c] != "empty":
                    occupied.add((r, c))

        for obj_name, obj_count in self._objects.items():
            if obj_name in ["altar", "generator"]:
                continue  # Already placed strategically

            current_count = np.sum(grid == obj_name)
            remaining_count = obj_count - current_count

            for _ in range(remaining_count):
                pos = self._get_random_empty_position(grid, occupied)
                if pos is not None:
                    r, c = pos
                    grid[r, c] = obj_name
                    occupied.add(pos)

    def _get_random_empty_position(self, grid: np.ndarray, occupied: set) -> Optional[Tuple[int, int]]:
        """Get a random empty position not in occupied set."""
        empty_positions = [(r, c) for r in range(self._height)
                          for c in range(self._width)
                          if grid[r, c] == "empty" and (r, c) not in occupied]

        if empty_positions:
            return tuple(self._rng.choice(empty_positions))
        return None
