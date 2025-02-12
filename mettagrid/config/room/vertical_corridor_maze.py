from mettagrid.config.room.room import Room
import numpy as np

class VerticalCorridorMaze(Room):
    def __init__(self,
                 width: int,
                 height: int,
                 border_width: int = 1,
                 corridor_width: int = 2,
                 arm_length: int = 10,
                 num_generators: int = 0,
                 num_convertors: int = 0,
                 num_heart_altars: int = 0,
                 seed = None,
                 agents: int = 1,
                 **kwargs):
        """
        Creates a maze with a vertical corridor down the middle and a variable number of horizontal arms.
        
        The grid is filled with walls. A vertical corridor (of width corridor_width) is carved down the center.
        Then, a number of horizontal arms are evenly spaced along the corridor. The total number of arms is the sum of:
          num_generators + num_convertors + num_heart_altars.
        Arms alternate direction: even-indexed arms extend left and odd-indexed arms extend right.
        
        At the end of each arm, a resource is placed. The first num_convertors arms (in resource order)
        will get a "convertor", the next num_heart_altars arms get a "heart altar", and the final num_generators arms
        get a "generator".
        
        The agent is placed near the top of the vertical corridor.
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
        self.agents = agents if isinstance(agents, int) else agents
        self._rng = np.random.default_rng(seed)
        # Initialize the grid with walls.
        self._grid = np.full((self.height, self.width), "wall", dtype='<U50')

    def _build(self) -> np.ndarray:
        grid = self._grid.copy()
        mid_x = self.width // 2

        # Determine horizontal bounds of the vertical corridor.
        half_corr = self.corridor_width // 2
        if self.corridor_width % 2 == 0:
            v_left = mid_x - half_corr
            v_right = mid_x + half_corr - 1
        else:
            v_left = mid_x - half_corr
            v_right = mid_x + half_corr

        # Carve the vertical corridor (entire height between v_left and v_right).
        for y in range(0, self.height):
            for x in range(v_left, v_right + 1):
                grid[y, x] = "empty"

        # Evenly space the arms along the vertical corridor.
        arm_y_positions = []
        spacing = (self.height - 2 * self.border_width) / (self.num_arms + 1)
        for i in range(self.num_arms):
            arm_y = int(self.border_width + (i + 1) * spacing)
            arm_y_positions.append(arm_y)

        # Determine arm directions: alternate "left" and "right".
        arm_directions = []
        for i in range(self.num_arms):
            arm_directions.append("left" if i % 2 == 0 else "right")

        # Build the resource list based on the parameters.
        resource_list = (["converter"] * self.num_convertors +
                         ["altar"] * self.num_heart_altars +
                         ["generator"] * self.num_generators)
        # If there are fewer resources than arms (shouldn't happen if parameters are set correctly),
        # then fill the rest with "empty".
        if len(resource_list) < self.num_arms:
            resource_list += ["empty"] * (self.num_arms - len(resource_list))
        
        # Shuffle the resource list to randomize placement
        self._rng.shuffle(resource_list)
        
        # Carve each horizontal arm and place its resource.
        for i, arm_y in enumerate(arm_y_positions):
            # Carve a horizontal corridor of thickness corridor_width centered at arm_y.
            half_thick = self.corridor_width // 2
            if self.corridor_width % 2 == 0:
                arm_y_start = arm_y - half_thick
                arm_y_end = arm_y + half_thick - 1
            else:
                arm_y_start = arm_y - half_thick
                arm_y_end = arm_y + half_thick

            direction = arm_directions[i]
            resource = resource_list[i]
            if direction == "left":
                # Extend arm leftwards from the left boundary of the vertical corridor.
                start_x = v_left
                end_x = max(self.border_width, start_x - self.arm_length)
                for y in range(arm_y_start, arm_y_end + 1):
                    for x in range(end_x, start_x):
                        grid[y, x] = "empty"
                # Place resource at the far left end.
                grid[arm_y, end_x] = resource
            else:  # direction == "right"
                # Extend arm rightwards from the right boundary of the vertical corridor.
                start_x = v_right
                end_x = min(self.width - self.border_width - 1, start_x + self.arm_length)
                for y in range(arm_y_start, arm_y_end + 1):
                    for x in range(start_x + 1, end_x + 1):
                        grid[y, x] = "empty"
                # Place resource at the far right end.
                grid[arm_y, end_x] = resource

        # Place the agent near the top of the vertical corridor.
        if grid[self.border_width, mid_x] == "empty":
            grid[self.border_width, mid_x] = "agent.agent"
        else:
            # Fallback: search downward for the first empty cell.
            placed = False
            for y in range(self.border_width, self.height):
                for x in range(self.width):
                    if grid[y, x] == "empty":
                        grid[y, x] = "agent.agent"
                        placed = True
                        break
                if placed:
                    break

        return grid
