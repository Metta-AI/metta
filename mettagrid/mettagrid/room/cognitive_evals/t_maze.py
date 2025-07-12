"""
Defines the TMaze room environment.

This environment creates a T-shaped maze with configurable dimensions and orientation.
It's designed for memory tasks where an agent starts at one end of the T, observes
a mine that indicates the correct direction, and must navigate to the corresponding
generator at the end of one of the T's arms.

The T-maze has three main components:
- Main hall: The "stem" of the T where the agent starts
- Two side halls: The "arms" of the T where generators are placed
- Variable orientation: T can point up, down, left, or right

Objects placed:
- One mine at the starting position (indicates reward direction via color)
- Two generators at the ends of the side halls
- Agent starts at the same position as the mine
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig

from mettagrid.room.room import Room


# Class definition for a T-shaped maze environment
class TMaze(Room):
    def __init__(
        self,
        main_hall_length: int = 5,  # Length of the main vertical/horizontal hall
        side_halls_length: int = 3,  # Length of the side arms
        hall_width: int = 1,  # Width of all halls
        orientation: str = "up",  # Direction the T points: "up", "down", "left", "right"
        mine_color: str = "red",  # Color of the mine object at start
        generator_color_left: str = "red",  # Color of the left/up generator
        generator_color_right: str = "blue",  # Color of the right/down generator
        agents: Union[int, DictConfig] = 1,  # Number or config of agents to place
        border_width: int = 0,  # Width of wall border around maze
        border_object: str = "wall",  # Object type for border (configurable for multi-room scenarios)
        seed: Optional[int] = None,  # Random seed for reproducibility
    ):
        # Initialize random number generator with optional seed
        self._rng = np.random.default_rng(seed)
        # Store border width
        self._border_width = border_width

        # Validate input parameters
        if main_hall_length < 2:
            raise ValueError(f"Main hall length must be at least 2, got {main_hall_length}")
        if side_halls_length < 1:
            raise ValueError(f"Side halls length must be at least 1, got {side_halls_length}")
        if hall_width < 2:
            raise ValueError(f"Hall width must be at least 2, got {hall_width}")
        if orientation not in ["up", "down", "left", "right"]:
            raise ValueError(f"Orientation must be one of ['up', 'down', 'left', 'right'], got {orientation}")

        # Store validated parameters
        self._main_hall_length = main_hall_length
        self._side_halls_length = side_halls_length
        self._hall_width = hall_width
        self._orientation = orientation
        self._mine_color = mine_color
        self._generator_color_left = generator_color_left
        self._generator_color_right = generator_color_right

        # Validate and store agents configuration
        if isinstance(agents, int):
            if agents < 0:
                raise ValueError("Number of agents cannot be negative.")
        elif isinstance(agents, DictConfig):
            for agent_name, count_val in agents.items():
                if not isinstance(count_val, int) or count_val < 0:
                    raise ValueError(
                        f"Agent count for '{str(agent_name)}' must be a non-negative integer, got {count_val}"
                    )
        else:
            raise TypeError(f"Agents parameter must be an int or a DictConfig, got {type(agents)}")
        self._agents_spec = agents

        # Calculate grid dimensions based on orientation
        if orientation in ["up", "down"]:
            # T extends horizontally, main hall is vertical
            core_width = 2 * side_halls_length + hall_width
            core_height = main_hall_length + hall_width
        else:  # left or right
            # T extends vertically, main hall is horizontal
            core_width = main_hall_length + hall_width
            core_height = 2 * side_halls_length + hall_width

        # Add border to core dimensions
        actual_grid_width = core_width + 2 * self._border_width
        actual_grid_height = core_height + 2 * self._border_width

        # Initialize parent Room class with border and labels
        super().__init__(
            border_width=self._border_width,
            border_object=border_object,
            labels=[
                "t_maze",
                orientation,
                f"main_{main_hall_length}",
                f"sides_{side_halls_length}",
                f"width_{hall_width}",
                f"mine_{mine_color}",
                f"generator_left_{generator_color_left}",
                f"generator_right_{generator_color_right}",
            ],
        )

        # Store grid dimensions
        self._width = actual_grid_width
        self._height = actual_grid_height

        # Initialize occupancy grid (all cells initially occupied)
        self._occ = np.zeros((self._height, self._width), dtype=bool)
        self.set_size_labels(self._width, self._height)

    def _build(self) -> np.ndarray:
        # Create grid filled with walls
        grid = np.full((self._height, self._width), "wall", dtype=object)
        self._occ.fill(True)

        # Get maze layout based on orientation
        if self._orientation == "up":
            # Main hall goes up, arms extend left and right
            start_pos, left_gen_pos, right_gen_pos, hall_cells = self._build_up_orientation()
        elif self._orientation == "down":
            # Main hall goes down, arms extend left and right
            start_pos, left_gen_pos, right_gen_pos, hall_cells = self._build_down_orientation()
        elif self._orientation == "left":
            # Main hall goes left, arms extend up and down
            start_pos, up_gen_pos, down_gen_pos, hall_cells = self._build_left_orientation()
            left_gen_pos, right_gen_pos = up_gen_pos, down_gen_pos  # Map to consistent naming
        else:  # right
            # Main hall goes right, arms extend up and down
            start_pos, up_gen_pos, down_gen_pos, hall_cells = self._build_right_orientation()
            left_gen_pos, right_gen_pos = up_gen_pos, down_gen_pos  # Map to consistent naming

        # Create empty spaces for the maze halls
        for r, c in hall_cells:
            if 0 <= r < self._height and 0 <= c < self._width:
                grid[r, c] = "empty"
                self._occ[r, c] = False

        # Place mine at starting position
        if 0 <= start_pos[0] < self._height and 0 <= start_pos[1] < self._width:
            grid[start_pos] = f"mine.{self._mine_color}"
            self._occ[start_pos] = True

        # Place generators at the ends of side halls
        grid[left_gen_pos] = f"generator.{self._generator_color_left}"
        grid[right_gen_pos] = f"generator.{self._generator_color_right}"

        # Mark generator positions as occupied
        self._occ[left_gen_pos] = True
        self._occ[right_gen_pos] = True

        # Place agent(s) near the starting position
        self._place_agents(grid, start_pos, hall_cells)

        return grid

    def _build_up_orientation(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]:
        """Build T-maze pointing upward. Returns start_pos, left_gen_pos, right_gen_pos, hall_cells."""
        hall_cells = []

        # Calculate key positions
        junction_r = self._border_width  # Junction at top
        junction_c = self._border_width + self._side_halls_length  # Center of junction
        start_r = self._border_width + self._main_hall_length - 1  # Start at bottom

        # Create main vertical hall
        for i in range(self._main_hall_length):
            for w in range(self._hall_width):
                r = start_r - i
                c = junction_c + w
                hall_cells.append((r, c))

        # Create left horizontal arm
        for i in range(self._side_halls_length):
            for w in range(self._hall_width):
                r = junction_r + w
                c = junction_c - 1 - i
                hall_cells.append((r, c))

        # Create right horizontal arm
        for i in range(self._side_halls_length):
            for w in range(self._hall_width):
                r = junction_r + w
                c = junction_c + self._hall_width + i
                hall_cells.append((r, c))

        # Calculate final positions
        start_pos = (start_r, junction_c)
        left_gen_pos = (junction_r, junction_c - self._side_halls_length)
        right_gen_pos = (junction_r, junction_c + self._hall_width + self._side_halls_length - 1)

        return start_pos, left_gen_pos, right_gen_pos, hall_cells

    def _build_down_orientation(
        self,
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]:
        """Build T-maze pointing downward. Returns start_pos, left_gen_pos, right_gen_pos, hall_cells."""
        hall_cells = []

        # Calculate key positions
        junction_r = self._border_width + self._main_hall_length - 1  # Junction at bottom
        junction_c = self._border_width + self._side_halls_length  # Center of junction
        start_r = self._border_width  # Start at top

        # Create main vertical hall
        for i in range(self._main_hall_length):
            for w in range(self._hall_width):
                r = start_r + i
                c = junction_c + w
                hall_cells.append((r, c))

        # Left arm (horizontal, going left from junction)
        for i in range(self._side_halls_length):
            for w in range(self._hall_width):
                r = junction_r + w
                c = junction_c - 1 - i
                hall_cells.append((r, c))

        # Right arm (horizontal, going right from junction)
        for i in range(self._side_halls_length):
            for w in range(self._hall_width):
                r = junction_r + w
                c = junction_c + self._hall_width + i
                hall_cells.append((r, c))

        # Positions
        start_pos = (start_r, junction_c)
        left_gen_pos = (junction_r, junction_c - self._side_halls_length)
        right_gen_pos = (junction_r, junction_c + self._hall_width + self._side_halls_length - 1)

        return start_pos, left_gen_pos, right_gen_pos, hall_cells

    def _build_left_orientation(
        self,
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]:
        """Build T-maze pointing leftward. Returns start_pos, up_gen_pos, down_gen_pos, hall_cells."""
        hall_cells = []

        # Calculate positions - junction at left, start at right
        junction_r = self._border_width + self._side_halls_length
        junction_c = self._border_width
        start_c = self._border_width + self._main_hall_length - 1

        # Main hall (horizontal, from start left to junction)
        for i in range(self._main_hall_length):
            for w in range(self._hall_width):
                r = junction_r + w
                c = start_c - i
                hall_cells.append((r, c))

        # Up arm (vertical, going up from junction)
        for i in range(self._side_halls_length):
            for w in range(self._hall_width):
                r = junction_r - 1 - i
                c = junction_c + w
                hall_cells.append((r, c))

        # Down arm (vertical, going down from junction)
        for i in range(self._side_halls_length):
            for w in range(self._hall_width):
                r = junction_r + self._hall_width + i
                c = junction_c + w
                hall_cells.append((r, c))

        # Positions
        start_pos = (junction_r, start_c)
        up_gen_pos = (junction_r - self._side_halls_length, junction_c)
        down_gen_pos = (junction_r + self._hall_width + self._side_halls_length - 1, junction_c)

        return start_pos, up_gen_pos, down_gen_pos, hall_cells

    def _build_right_orientation(
        self,
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]:
        """Build T-maze pointing rightward. Returns start_pos, up_gen_pos, down_gen_pos, hall_cells."""
        hall_cells = []

        # Calculate positions - junction at right, start at left
        junction_r = self._border_width + self._side_halls_length
        junction_c = self._border_width + self._main_hall_length - 1
        start_c = self._border_width

        # Main hall (horizontal, from start right to junction)
        for i in range(self._main_hall_length):
            for w in range(self._hall_width):
                r = junction_r + w
                c = start_c + i
                hall_cells.append((r, c))

        # Up arm (vertical, going up from junction)
        for i in range(self._side_halls_length):
            for w in range(self._hall_width):
                r = junction_r - 1 - i
                c = junction_c + w
                hall_cells.append((r, c))

        # Down arm (vertical, going down from junction)
        for i in range(self._side_halls_length):
            for w in range(self._hall_width):
                r = junction_r + self._hall_width + i
                c = junction_c + w
                hall_cells.append((r, c))

        # Positions
        start_pos = (junction_r, start_c)
        up_gen_pos = (junction_r - self._side_halls_length, junction_c)
        down_gen_pos = (junction_r + self._hall_width + self._side_halls_length - 1, junction_c)

        return start_pos, up_gen_pos, down_gen_pos, hall_cells

    def _place_agents(self, grid: np.ndarray, start_pos: Tuple[int, int], hall_cells: List[Tuple[int, int]]):
        """Place agent(s) in the maze, preferably near the starting position."""
        agent_symbols_to_place = []

        if isinstance(self._agents_spec, int):
            count = self._agents_spec
            agent_symbols_to_place = ["agent.agent"] * count
        elif isinstance(self._agents_spec, DictConfig):
            for name, count_val in self._agents_spec.items():
                s_name = str(name)
                processed_name = s_name if "." in s_name else f"agent.{s_name}"
                agent_symbols_to_place.extend([processed_name] * count_val)
            self._rng.shuffle(agent_symbols_to_place)

        # Find empty cells near the starting position for agent placement
        empty_hall_cells = [pos for pos in hall_cells if grid[pos] == "empty"]
        self._rng.shuffle(empty_hall_cells)

        # Try to place agents, starting with cells closest to start_pos
        start_r, start_c = start_pos
        empty_hall_cells.sort(key=lambda pos: abs(pos[0] - start_r) + abs(pos[1] - start_c))

        agents_placed = 0
        for i in range(min(len(agent_symbols_to_place), len(empty_hall_cells))):
            pos = empty_hall_cells[i]
            grid[pos] = agent_symbols_to_place[i]
            self._occ[pos] = True
            agents_placed += 1

        if agents_placed < len(agent_symbols_to_place):
            print(f"Warning: Could only place {agents_placed}/{len(agent_symbols_to_place)} agents in T-maze.")
