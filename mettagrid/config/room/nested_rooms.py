from typing import Optional, Tuple, List
import numpy as np
from omegaconf import DictConfig
from mettagrid.config.room.room import Room

class NestedRoomsMaze(Room):
    """
    An environment with nested rooms (rooms within rooms). Each room is drawn with walls on all four sides,
    except that one randomly chosen wall gets a door (a contiguous gap of cells of length door_size).
    The overall interior is split into nested rooms by repeatedly inseting a fixed margin.
    
    **Reward Placement:**  
      - The **agent** is placed in the innermost room.  
      - The **generator** is placed in the second innermost room (if available; otherwise in the innermost).  
      - The **converter** is placed in a room roughly in the middle of the nesting.  
      - The **heart altar** is placed in the outermost room.
      
    Doors remain at random locations for each room.
    """
    def __init__(
        self,
        width: int,
        height: int,
        num_rooms: int,
        door_size: int,
        agent_in_innermost: bool = True,
        agents: int | DictConfig = 1,
        seed: Optional[int] = None,
        border_width: int = 1,
        border_object: str = "wall",
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._width = width
        self._height = height
        self._num_rooms = num_rooms
        self._door_size = door_size
        self._agent_in_innermost = agent_in_innermost
        self._agents = agents
        self._rng = np.random.default_rng(seed)
        self._border_width = border_width
        self._border_object = border_object

        # Initialize the grid with "empty" cells.
        self._grid = np.full((height, width), "empty", dtype='<U50')

    def _build(self) -> np.ndarray:
        # --- Draw the Outer Border ---
        for y in range(0, self._border_width):
            self._grid[y, :] = self._border_object
        for y in range(self._height - self._border_width, self._height):
            self._grid[y, :] = self._border_object
        for x in range(0, self._border_width):
            self._grid[:, x] = self._border_object
        for x in range(self._width - self._border_width, self._width):
            self._grid[:, x] = self._border_object

        # --- Define the Overall Interior (inside the outer border) ---
        interior_x0 = self._border_width
        interior_y0 = self._border_width
        interior_x1 = self._width - self._border_width - 1
        interior_y1 = self._height - self._border_width - 1

        # --- Compute Nested Room Boundaries ---
        # Use a fixed margin (gap) between consecutive rooms.
        margin = 3
        rooms = []  # Each room is defined as (rx0, ry0, rx1, ry1) for its wall boundaries.
        for i in range(self._num_rooms):
            rx0 = interior_x0 + i * margin
            ry0 = interior_y0 + i * margin
            rx1 = interior_x1 - i * margin
            ry1 = interior_y1 - i * margin
            if rx0 >= rx1 or ry0 >= ry1:
                break
            rooms.append((rx0, ry0, rx1, ry1))
        
        # --- Draw Walls for Each Nested Room with a Random Door ---
        # For each room, choose a random wall to have a door gap.
        # (We record the door side for the outermost room in case it is useful.)
        outer_door_side = None  
        for idx, (rx0, ry0, rx1, ry1) in enumerate(rooms):
            door_side = self._rng.choice(["top", "bottom", "left", "right"])
            if idx == 0:
                outer_door_side = door_side
            # Top wall.
            if door_side == "top":
                door_x_start = int(self._rng.integers(low=rx0, high=rx1 - self._door_size + 2))
                door_x_end = door_x_start + self._door_size
                for x in range(rx0, rx1 + 1):
                    if not (door_x_start <= x < door_x_end):
                        self._grid[ry0, x] = self._border_object
            else:
                for x in range(rx0, rx1 + 1):
                    self._grid[ry0, x] = self._border_object

            # Bottom wall.
            if door_side == "bottom":
                door_x_start = int(self._rng.integers(low=rx0, high=rx1 - self._door_size + 2))
                door_x_end = door_x_start + self._door_size
                for x in range(rx0, rx1 + 1):
                    if not (door_x_start <= x < door_x_end):
                        self._grid[ry1, x] = self._border_object
            else:
                for x in range(rx0, rx1 + 1):
                    self._grid[ry1, x] = self._border_object

            # Left wall.
            if door_side == "left":
                door_y_start = int(self._rng.integers(low=ry0, high=ry1 - self._door_size + 2))
                door_y_end = door_y_start + self._door_size
                for y in range(ry0, ry1 + 1):
                    if not (door_y_start <= y < door_y_end):
                        self._grid[y, rx0] = self._border_object
            else:
                for y in range(ry0, ry1 + 1):
                    self._grid[y, rx0] = self._border_object

            # Right wall.
            if door_side == "right":
                door_y_start = int(self._rng.integers(low=ry0, high=ry1 - self._door_size + 2))
                door_y_end = door_y_start + self._door_size
                for y in range(ry0, ry1 + 1):
                    if not (door_y_start <= y < door_y_end):
                        self._grid[y, rx1] = self._border_object
            else:
                for y in range(ry0, ry1 + 1):
                    self._grid[y, rx1] = self._border_object

        # --- Helper to Compute the Interior of a Room ---
        # (One cell inset from its walls.)
        def room_interior(room: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
            rx0, ry0, rx1, ry1 = room
            return (rx0 + 1, ry0 + 1, rx1 - 1, ry1 - 1)

        # --- Place the Agent and Rewards in Different Nested Rooms ---
        # We use the following assignments if enough rooms exist:
        #   - Agent: innermost room (rooms[-1])
        #   - Generator: second innermost (rooms[-2] if available; else innermost)
        #   - Converter: median room (rooms[len(rooms)//2] if at least 3 exist; else innermost)
        #   - Heart Altar: outermost room (rooms[0])
        if len(rooms) == 0:
            # No rooms were created—fallback: place everything at center.
            cx, cy = self._width // 2, self._height // 2
            self._grid[cy, cx] = "agent.agent"
            self._grid[cy, cx-1] = "generator"
            self._grid[cy, cx] = "converter"
            self._grid[cy, cx+1] = "altar"
            return self._grid

        # Agent in innermost room.
        agent_room = rooms[-1]
        a_rx0, a_ry0, a_rx1, a_ry1 = room_interior(agent_room)
        agent_x = (a_rx0 + a_rx1) // 2
        agent_y = (a_ry0 + a_ry1) // 2
        self._grid[agent_y, agent_x] = "agent.agent"

        # Generator in second innermost room, if available.
        if len(rooms) >= 2:
            generator_room = rooms[-2]
        else:
            generator_room = agent_room
        g_rx0, g_ry0, g_rx1, g_ry1 = room_interior(generator_room)
        gen_x = (g_rx0 + g_rx1) // 2
        gen_y = (g_ry0 + g_ry1) // 2
        self._grid[gen_y, gen_x] = "generator"

        # Converter in the median room (if at least 3 rooms exist).
        if len(rooms) >= 3:
            median_index = len(rooms) // 2
            converter_room = rooms[median_index]
        else:
            converter_room = agent_room
        c_rx0, c_ry0, c_rx1, c_ry1 = room_interior(converter_room)
        conv_x = (c_rx0 + c_rx1) // 2
        conv_y = (c_ry0 + c_ry1) // 2
        self._grid[conv_y, conv_x] = "converter"

        # Heart Altar in the outermost room.
        altar_room = rooms[0]
        a_rx0, a_ry0, a_rx1, a_ry1 = room_interior(altar_room)
        altar_x = (a_rx0 + a_rx1) // 2
        altar_y = (a_ry0 + a_ry1) // 2
        self._grid[altar_y, altar_x] = "altar"

        return self._grid

    def _sample_reward_positions(
        self, interior: Tuple[int, int, int, int], num: int, min_distance: int = 2
    ) -> List[Tuple[int, int]]:
        """
        (Unused in the updated reward placement.)
        """
        rx0, ry0, rx1, ry1 = interior
        positions = []
        attempts = 0
        max_attempts = 1000
        while len(positions) < num and attempts < max_attempts:
            attempts += 1
            x = int(self._rng.integers(rx0, rx1 + 1))
            y = int(self._rng.integers(ry0, ry1 + 1))
            candidate = (x, y)
            if all((abs(x - ox) + abs(y - oy)) >= min_distance for (ox, oy) in positions):
                positions.append(candidate)
        if len(positions) < num:
            center = ((rx0 + rx1) // 2, (ry0 + ry1) // 2)
            while len(positions) < num:
                positions.append(center)
        return positions
