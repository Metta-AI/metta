"""
MinGrid-Inspired Multi-Room Navigation Environment

Inspired by MinGrid's MultiRoomEnv and FourRoomsEnv. This creates multiple connected rooms
with wall barriers between them. Each room can have different objectives and resources,
requiring agents to explore and navigate between rooms.

Original MinGrid MultiRoomEnv: Multiple rooms connected by doors, each with different objectives.
MettagGrid adaptation: Multiple rooms connected by gaps in walls, with altars/generators in different rooms.

The environment consists of:
- Multiple rectangular rooms separated by walls
- Gaps between rooms for navigation (instead of doors)
- Different resources/objectives in each room
- Agents must explore multiple rooms to complete tasks
"""

from typing import Optional, Tuple, List
import numpy as np
from omegaconf import DictConfig

from mettagrid.room.room import Room


class MinigridMultiRoomNavigation(Room):
    def __init__(
        self,
        width: int,
        height: int,
        objects: DictConfig,
        agents: int | DictConfig = 1,
        seed: Optional[int] = None,
        border_width: int = 1,
        border_object: str = "wall",
        num_rooms_x: int = 2,  # Number of rooms horizontally
        num_rooms_y: int = 2,  # Number of rooms vertically
        room_gap_size: int = 3,  # Size of gaps between rooms
        min_room_size: int = 5,  # Minimum room dimension
        team: str | None = None,
    ):
        super().__init__(border_width=border_width, border_object=border_object, labels=["minigrid_multi_room"])
        self.set_size_labels(width, height)
        self._rng = np.random.default_rng(seed)
        self._width = width
        self._height = height
        self._agents = agents
        self._objects = objects
        self._num_rooms_x = num_rooms_x
        self._num_rooms_y = num_rooms_y
        self._room_gap_size = room_gap_size
        self._min_room_size = min_room_size
        self._team = team

    def _build(self) -> np.ndarray:
        # Create empty grid
        grid = np.full((self._height, self._width), "empty", dtype="<U50")

        # Calculate room layout
        room_layout = self._calculate_room_layout()

        # Create room walls
        self._create_room_walls(grid, room_layout)

        # Place resources in different rooms
        self._place_resources_in_rooms(grid, room_layout)

        # Place agents
        self._place_agents_in_rooms(grid, room_layout)

        return grid

    def _calculate_room_layout(self) -> List[List[Tuple[int, int, int, int]]]:
        """Calculate the position and size of each room. Returns [row][col] of (top, left, bottom, right)."""
        available_width = self._width - (self._num_rooms_x - 1) * self._room_gap_size
        available_height = self._height - (self._num_rooms_y - 1) * self._room_gap_size

        room_width = max(self._min_room_size, available_width // self._num_rooms_x)
        room_height = max(self._min_room_size, available_height // self._num_rooms_y)

        layout = []
        current_y = 0

        for row in range(self._num_rooms_y):
            room_row = []
            current_x = 0

            for col in range(self._num_rooms_x):
                # Calculate room bounds
                top = current_y
                left = current_x
                bottom = min(current_y + room_height, self._height)
                right = min(current_x + room_width, self._width)

                room_row.append((top, left, bottom, right))
                current_x = right + self._room_gap_size

            layout.append(room_row)
            current_y += room_height + self._room_gap_size

        return layout

    def _create_room_walls(self, grid: np.ndarray, room_layout: List[List[Tuple[int, int, int, int]]]) -> None:
        """Create walls around each room with gaps for navigation."""
        for row in range(len(room_layout)):
            for col in range(len(room_layout[row])):
                top, left, bottom, right = room_layout[row][col]

                # Create room walls
                # Top wall
                for c in range(left, right):
                    if top > 0:
                        grid[top, c] = "wall"

                # Bottom wall
                for c in range(left, right):
                    if bottom < self._height:
                        grid[bottom - 1, c] = "wall"

                # Left wall
                for r in range(top, bottom):
                    if left > 0:
                        grid[r, left] = "wall"

                # Right wall
                for r in range(top, bottom):
                    if right < self._width:
                        grid[r, right - 1] = "wall"

        # Create gaps between rooms
        self._create_room_gaps(grid, room_layout)

    def _create_room_gaps(self, grid: np.ndarray, room_layout: List[List[Tuple[int, int, int, int]]]) -> None:
        """Create gaps in walls to connect rooms."""
        for row in range(len(room_layout)):
            for col in range(len(room_layout[row])):
                top, left, bottom, right = room_layout[row][col]

                # Create gap to right room
                if col + 1 < len(room_layout[row]):
                    next_room = room_layout[row][col + 1]
                    gap_start = (top + bottom) // 2 - 1
                    gap_end = gap_start + 3  # 3-cell wide gap

                    for r in range(max(top, gap_start), min(bottom, gap_end)):
                        if right < self._width:
                            grid[r, right - 1] = "empty"  # Remove wall

                # Create gap to bottom room
                if row + 1 < len(room_layout):
                    next_room = room_layout[row + 1][col]
                    gap_start = (left + right) // 2 - 1
                    gap_end = gap_start + 3  # 3-cell wide gap

                    for c in range(max(left, gap_start), min(right, gap_end)):
                        if bottom < self._height:
                            grid[bottom - 1, c] = "empty"  # Remove wall

    def _place_resources_in_rooms(self, grid: np.ndarray, room_layout: List[List[Tuple[int, int, int, int]]]) -> None:
        """Place different resources in different rooms."""
        total_rooms = len(room_layout) * len(room_layout[0])

        # Distribute resources across rooms
        altar_count = self._objects.get("altar", 0)
        generator_count = self._objects.get("generator", 0)

        # Create resource distribution plan
        resource_plan = []

        # Distribute altars
        for i in range(altar_count):
            room_idx = i % total_rooms
            resource_plan.append(("altar", room_idx))

        # Distribute generators
        for i in range(generator_count):
            room_idx = (i + altar_count) % total_rooms
            resource_plan.append(("generator", room_idx))

        # Place resources
        for resource_type, room_idx in resource_plan:
            room_row = room_idx // len(room_layout[0])
            room_col = room_idx % len(room_layout[0])

            if room_row < len(room_layout) and room_col < len(room_layout[room_row]):
                self._place_resource_in_room(grid, room_layout[room_row][room_col], resource_type)

        # Place other objects
        for obj_name, obj_count in self._objects.items():
            if obj_name in ["altar", "generator"]:
                continue

            for i in range(obj_count):
                room_idx = i % total_rooms
                room_row = room_idx // len(room_layout[0])
                room_col = room_idx % len(room_layout[0])

                if room_row < len(room_layout) and room_col < len(room_layout[room_row]):
                    self._place_resource_in_room(grid, room_layout[room_row][room_col], obj_name)

    def _place_resource_in_room(self, grid: np.ndarray, room_bounds: Tuple[int, int, int, int],
                               resource_type: str) -> None:
        """Place a single resource in the specified room."""
        top, left, bottom, right = room_bounds

        # Find empty positions in room (avoiding walls)
        empty_positions = []
        for r in range(top + 1, bottom - 1):
            for c in range(left + 1, right - 1):
                if r < self._height and c < self._width and grid[r, c] == "empty":
                    empty_positions.append((r, c))

        if empty_positions:
            pos = tuple(self._rng.choice(empty_positions))
            r, c = pos
            grid[r, c] = resource_type

    def _place_agents_in_rooms(self, grid: np.ndarray, room_layout: List[List[Tuple[int, int, int, int]]]) -> None:
        """Place agents in starting room(s)."""
        if not room_layout or not room_layout[0]:
            return

        # Place agents in first room (top-left)
        start_room = room_layout[0][0]
        if isinstance(self._agents, int):
            if self._team is None:
                agents = ["agent.agent"] * self._agents
            else:
                agents = [f"agent.{self._team}"] * self._agents
        elif isinstance(self._agents, DictConfig):
            agents = [f"agent.{agent}" for agent, na in self._agents.items() for _ in range(na)]

        for i, agent in enumerate(agents):
            if i == 0:
                # Place first agent in starting room
                self._place_agent_in_room(grid, start_room, agent)
            else:
                # Place additional agents in other rooms if available
                total_rooms = len(room_layout) * len(room_layout[0])
                if i < total_rooms:
                    room_row = i // len(room_layout[0])
                    room_col = i % len(room_layout[0])
                    if room_row < len(room_layout) and room_col < len(room_layout[room_row]):
                        self._place_agent_in_room(grid, room_layout[room_row][room_col], agent)

    def _place_agent_in_room(self, grid: np.ndarray, room_bounds: Tuple[int, int, int, int], agent: str) -> None:
        """Place a single agent in the specified room."""
        top, left, bottom, right = room_bounds

        # Find empty positions in room
        empty_positions = []
        for r in range(top + 1, bottom - 1):
            for c in range(left + 1, right - 1):
                if r < self._height and c < self._width and grid[r, c] == "empty":
                    empty_positions.append((r, c))

        if empty_positions:
            pos = tuple(self._rng.choice(empty_positions))
            r, c = pos
            grid[r, c] = agent
