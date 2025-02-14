from mettagrid.config.room.room import Room
import numpy as np

class CorridorMaze(Room):
    def __init__(self,
                 width: int,
                 height: int,
                 border_width: int = 1,
                 corridor_width: int = 2,
                 arm_length: int = 10,
                 num_generators: int = 0,
                 num_convertors: int = 0,
                 num_heart_altars: int = 0,
                 seed=None,
                 agents: int = 1,
                 rotate: bool = False,
                 **kwargs):
        """
        Creates a maze with a vertical corridor down the middle and horizontal arms.
        The arms are evenly spaced along the corridor and alternate extending left and right.
        Resources (converter, altar, generator) are placed at the end of each arm,
        and the agent is placed near the top of the corridor.

        If rotate is True, the final maze is rotated by 90°.
        """
        super().__init__(border_width=border_width, border_object="wall")
        self.width = width
        self.height = height
        self.border_width = border_width
        self.corridor_width = corridor_width
        self.arm_length = arm_length
        self.num_generators = num_generators
        self.num_convertors = num_convertors
        self.num_heart_altars = num_heart_altars
        self.num_arms = num_generators + num_convertors + num_heart_altars
        self.seed = seed
        self.agents = agents
        self.rotate = rotate
        self._rng = np.random.default_rng(seed)
        self._grid = np.full((height, width), "wall", dtype='<U50')

    def _build(self) -> np.ndarray:
        grid = self._grid.copy()
        mid_x = self.width // 2

        # Determine the horizontal bounds of the vertical corridor.
        half_corr = self.corridor_width // 2
        if self.corridor_width % 2 == 0:
            v_left, v_right = mid_x - half_corr, mid_x + half_corr - 1
        else:
            v_left, v_right = mid_x - half_corr, mid_x + half_corr

        # Carve the vertical corridor via slicing.
        grid[:, v_left:v_right+1] = "empty"

        # Evenly space the arms along the corridor.
        spacing = (self.height - 2 * self.border_width) / (self.num_arms + 1)
        arm_y_positions = [int(self.border_width + (i + 1) * spacing) for i in range(self.num_arms)]
        arm_directions = ["left" if i % 2 == 0 else "right" for i in range(self.num_arms)]

        # Build and shuffle the resource list.
        resources = (["converter"] * self.num_convertors +
                     ["altar"] * self.num_heart_altars +
                     ["generator"] * self.num_generators)
        if len(resources) < self.num_arms:
            resources += ["empty"] * (self.num_arms - len(resources))
        self._rng.shuffle(resources)

        # Carve horizontal arms and place resources.
        for i, arm_y in enumerate(arm_y_positions):
            half_thick = self.corridor_width // 2
            if self.corridor_width % 2 == 0:
                arm_y_start, arm_y_end = arm_y - half_thick, arm_y + half_thick - 1
            else:
                arm_y_start, arm_y_end = arm_y - half_thick, arm_y + half_thick

            direction, resource = arm_directions[i], resources[i]
            if direction == "left":
                start_x = v_left
                end_x = max(self.border_width, start_x - self.arm_length)
                grid[arm_y_start:arm_y_end+1, end_x:start_x] = "empty"
                grid[arm_y, end_x] = resource
            else:
                start_x = v_right
                end_x = min(self.width - self.border_width - 1, start_x + self.arm_length)
                grid[arm_y_start:arm_y_end+1, start_x+1:end_x+1] = "empty"
                grid[arm_y, end_x] = resource

        # Place the agent near the top of the corridor.
        if grid[self.border_width, mid_x] == "empty":
            grid[self.border_width, mid_x] = "agent.agent"
        else:
            for y in range(self.border_width, self.height):
                for x in range(self.width):
                    if grid[y, x] == "empty":
                        grid[y, x] = "agent.agent"
                        break
                else:
                    continue
                break

        # Optionally rotate the maze 90°.
        if self.rotate:
            grid = np.rot90(grid)

        return grid
