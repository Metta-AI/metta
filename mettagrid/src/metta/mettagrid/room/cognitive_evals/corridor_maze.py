import numpy as np

from metta.mettagrid.room.room import Room
from metta.mettagrid.room.utils import create_grid


class CorridorMaze(Room):
    def __init__(
        self,
        width: int,
        height: int,
        border_width: int = 1,
        corridor_width: int = 2,
        arm_length: int = 10,
        num_mines: int = 0,
        num_convertors: int = 0,
        num_heart_altars: int = 0,
        seed=None,
        agents: int = 1,
        rotate: bool = False,
        **kwargs,
    ):
        """
        Maze with a vertical corridor and horizontal arms.
        Arms alternate left/right, with resources placed at their ends.
        Optionally rotates the maze 90°.
        """
        super().__init__(border_width=border_width, border_object="wall")
        self.width = width
        self.height = height
        self.border_width = border_width
        self.corridor_width = corridor_width
        self.arm_length = arm_length
        self.num_mines = num_mines
        self.num_convertors = num_convertors
        self.num_heart_altars = num_heart_altars
        self.num_arms = num_mines + num_convertors + num_heart_altars
        self.seed = seed
        self.agents = agents
        self.rotate = rotate
        self._rng = np.random.default_rng(seed)

    def _build(self) -> np.ndarray:
        # Use create_grid to initialize the grid (fill with "wall")
        grid = create_grid(self.height, self.width, fill_value="wall")
        mid_x = self.width // 2

        # Compute vertical corridor bounds.
        v_left = mid_x - (self.corridor_width // 2)
        v_right = v_left + self.corridor_width - 1
        grid[:, v_left : v_right + 1] = "empty"

        # Calculate evenly spaced arm positions along the corridor.
        spacing = (self.height - 2 * self.border_width) / (self.num_arms + 1)
        arm_y_positions = [int(self.border_width + (i + 1) * spacing) for i in range(self.num_arms)]
        directions = ["left" if i % 2 == 0 else "right" for i in range(self.num_arms)]

        # Build and shuffle the resource list.
        resources = ["generator"] * self.num_convertors + ["altar"] * self.num_heart_altars + ["mine"] * self.num_mines
        resources += ["empty"] * (self.num_arms - len(resources))
        self._rng.shuffle(resources)

        # Carve each horizontal arm and place its resource.
        for i, arm_y in enumerate(arm_y_positions):
            arm_top = arm_y - (self.corridor_width // 2)
            arm_bottom = arm_top + self.corridor_width - 1
            if directions[i] == "left":
                start_x = v_left
                end_x = max(self.border_width, start_x - self.arm_length)
                grid[arm_top : arm_bottom + 1, end_x:start_x] = "empty"
                grid[arm_y, end_x] = resources[i]
            else:
                start_x = v_right
                end_x = min(self.width - self.border_width - 1, start_x + self.arm_length)
                grid[arm_top : arm_bottom + 1, start_x + 1 : end_x + 1] = "empty"
                grid[arm_y, end_x] = resources[i]

        # Place the agent near the top of the corridor.
        if grid[self.border_width, mid_x] == "empty":
            grid[self.border_width, mid_x] = "agent.agent"
        else:
            for y in range(self.border_width, self.height):
                empty_positions = np.where(grid[y] == "empty")[0]
                if empty_positions.size:
                    grid[y, empty_positions[0]] = "agent.agent"
                    break

        if self.rotate:
            grid = np.rot90(grid)

        return grid
