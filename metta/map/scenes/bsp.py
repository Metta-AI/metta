import logging
from typing import Literal, Tuple

import numpy as np

from metta.common.util.config import Config
from metta.map.scene import Scene
from metta.map.types import MapGrid

logger = logging.getLogger(__name__)

Direction = Literal["horizontal", "vertical"]


class BSPLayoutParams(Config):
    area_count: int


class BSPLayout(Scene[BSPLayoutParams]):
    """
    This scene doesn't render anything, it just creates areas that can be used by other scenes.
    """

    def render(self):
        grid = self.grid

        tree = BSPTree(
            width=grid.shape[1],
            height=grid.shape[0],
            leaf_zone_count=self.params.area_count,
            rng=self.rng,
        )
        for zone in tree.get_leaf_zones():
            self.make_area(zone.x, zone.y, zone.width, zone.height, tags=["zone"])


class BSPParams(Config):
    rooms: int
    min_room_size: int
    min_room_size_ratio: float
    max_room_size_ratio: float
    skip_corridors: bool = False


class BSP(Scene[BSPParams]):
    """
    Binary Space Partitioning. (Roguelike dungeon generator)

    This scene creates a grid of rooms, and then connects them with corridors.
    """

    def render(self):
        grid = self.grid
        params = self.params

        grid[:] = "wall"

        bsp_tree = BSPTree(
            width=grid.shape[1],
            height=grid.shape[0],
            leaf_zone_count=params.rooms,
            rng=self.rng,
        )

        # Make rooms
        rooms: list[Zone] = []
        for zone in bsp_tree.get_leaf_zones():
            room = zone.make_room(
                min_size=params.min_room_size,
                min_size_ratio=params.min_room_size_ratio,
                max_size_ratio=params.max_room_size_ratio,
            )
            rooms.append(room)

            grid[room.y : room.y + room.height, room.x : room.x + room.width] = "empty"
            self.make_area(room.x, room.y, room.width, room.height, tags=["room"])

        # Make corridors
        if params.skip_corridors:
            logger.info("Skipping corridors")
            return

        for zone1, zone2 in bsp_tree.get_sibling_pairs():
            corridor_direction = "vertical" if zone1.x == zone2.x else "horizontal"

            used_grid = grid
            if corridor_direction == "horizontal":
                used_grid = np.transpose(grid)
                zone1 = zone1.transpose()
                zone2 = zone2.transpose()

            if zone1.y > zone2.y:
                (zone1, zone2) = (zone2, zone1)

            surface1 = Surface.from_zone(used_grid, zone1, "up")
            surface2 = Surface.from_zone(used_grid, zone2, "down")

            lines = connect_surfaces(surface1, surface2)

            if corridor_direction == "horizontal":
                lines = [line.transpose() for line in lines]

            # draw lines on the original grid
            for line in lines:
                if line.direction == "vertical":
                    grid[line.start[1] : line.start[1] + line.length, line.start[0]] = "empty"
                else:
                    grid[line.start[1], line.start[0] : line.start[0] + line.length] = "empty"


class Zone:
    def __init__(self, x: int, y: int, width: int, height: int, rng: np.random.Generator):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rng = rng

    def split(self) -> Tuple["Zone", "Zone"]:
        # Split in random direction, unless the room is too wide or too tall.
        if self.width > self.height * 2:
            # Note: vertical split means vertical line, not vertical layout
            direction = "vertical"
        elif self.height > self.width * 2:
            direction = "horizontal"
        else:
            direction = self.rng.choice(["horizontal", "vertical"])

        if direction == "horizontal":
            return self.horizontal_split()
        else:
            return self.vertical_split()

    def random_divide(self, size: int):
        min_size = size // 3  # TODO - configurable proportion
        return self.rng.integers(min_size, size - min_size + 1, dtype=int)

    def horizontal_split(self) -> Tuple["Zone", "Zone"]:
        first_height = self.random_divide(self.height)
        return (
            Zone(self.x, self.y, self.width, first_height, self.rng),
            Zone(
                self.x,
                self.y + first_height,
                self.width,
                self.height - first_height,
                self.rng,
            ),
        )

    def vertical_split(self) -> Tuple["Zone", "Zone"]:
        (child1, child2) = self.transpose().horizontal_split()
        return (child1.transpose(), child2.transpose())

    def make_room(
        self,
        min_size: int,
        min_size_ratio: float,
        max_size_ratio: float,
    ) -> "Zone":
        # Randomly determine room size
        def random_size(n: int) -> int:
            return self.rng.integers(
                max(min_size, int(n * min_size_ratio)),
                max(min_size, int(n * max_size_ratio)) + 1,
                dtype=int,
            )

        room_width = random_size(self.width)
        room_height = random_size(self.height)

        # Randomly position the room within the zone; always leave a 1 cell border on bottom-right, otherwise the
        # rooms could touch each other.
        shift_x = self.rng.integers(1, max(1, self.width - room_width) + 1, dtype=int)
        shift_y = self.rng.integers(1, max(1, self.height - room_height) + 1, dtype=int)
        return Zone(self.x + shift_x, self.y + shift_y, room_width, room_height, self.rng)

    def transpose(self) -> "Zone":
        # Zones can be transposed, to avoid having to write code for both horizontal and vertical splits.
        # See also: Line.transpose()
        return Zone(self.y, self.x, self.height, self.width, self.rng)

    def __repr__(self):
        return f"Zone({self.x}, {self.y}, {self.width}, {self.height})"


class Surface:
    """
    When choosing how to connect rooms, or rooms with corridors, we need to represent the surface of possible
    attachment points.

    Surface example:

    │#.........##│
    │###......###│
    │###......###│
    │############│
    │############│
    └────────────┘

    In this example, the surface is the set of . characters that can be approached from below (side="down").

    The empty areas on the left and right are not part of the surface.

    The code that uses the surface doesn't care what the surface is made of, it just needs to know which that can be
    approached.
    """

    def __init__(
        self,
        min_x: int,
        ys: list[int],
        side: Literal["up", "down"],
        rng: np.random.Generator,
    ):
        self.min_x = min_x
        self.ys = ys
        self.side = side
        self.rng = rng

    @property
    def max_y(self) -> int:
        return max(self.ys)

    @property
    def min_y(self) -> int:
        return min(self.ys)

    @property
    def max_x(self) -> int:
        # Last column of the surface
        return self.min_x + len(self.ys) - 1

    def random_position(self) -> Tuple[int, int]:
        # Choose a position from which we can draw a vertical corridor.
        valid_xs = []

        def behind(y1, y2) -> bool:
            if self.side == "up":
                return y1 > y2
            else:
                return y1 < y2

        for i, y in enumerate(self.ys):
            # We want to exclude the columns where the vertical line would be adjacent to the surface.
            if i > 0 and behind(y, self.ys[i - 1]):
                continue
            if i < len(self.ys) - 1 and behind(y, self.ys[i + 1]):
                continue

            valid_xs.append(i)

        x = self.rng.choice(valid_xs)
        return (x + self.min_x, self.ys[x])

    @staticmethod
    def from_zone(grid: MapGrid, zone: Zone, side: Literal["up", "down"]) -> "Surface":
        # Scan the entire zone, starting from the top or bottom, and collect all the y values that are part of
        # the surface.
        min_x = None
        ys: list[int] = []

        for x in range(zone.x, zone.x + zone.width):
            yrange = range(zone.y, zone.y + zone.height)
            if side == "up":
                yrange = reversed(yrange)

            y_value = None
            for y in yrange:
                if grid[y, x] == "empty":
                    y_value = y
                    break

            if y_value is None:
                # haven't started or already ended?
                if min_x is None:
                    # ok, haven't started
                    continue
                else:
                    # we're done
                    # TODO - assert that there are no breaks in the surface?
                    break
            else:
                if min_x is None:
                    min_x = x
                ys.append(y_value)

        if min_x is None:
            raise ValueError("No surface found")

        return Surface(min_x, ys, side, zone.rng)

    def __repr__(self):
        return f"Surface(min_x={self.min_x}, ys={self.ys})"


class Line:
    """
    A line is a straight corridor that can be drawn on the grid.

    It can be horizontal or vertical.

    Full corridor between two rooms can be represented as multiple lines.
    """

    def __init__(self, direction: Direction, start: Tuple[int, int], length: int):
        self.direction = direction

        if length < 0:
            # line of negative length means that the line is reversed (right to left or down to up)
            length = -length
            if direction == "horizontal":
                start = (start[0] - length + 1, start[1])
            else:
                start = (start[0], start[1] - length + 1)

        self.start = start
        self.length = length

    def transpose(self) -> "Line":
        # Trick to avoid having to write code for both horizontal and vertical lines.
        # See also: Zone.transpose()
        direction = "horizontal" if self.direction == "vertical" else "vertical"
        return Line(direction, (self.start[1], self.start[0]), self.length)

    def __repr__(self):
        return f"Line({self.direction}, {self.start}, {self.length})"


def connect_surfaces(surface1: Surface, surface2: Surface):
    """
    Connect two surfaces with a corridor.

    Assumes that the surfaces are adjacent and the surface1 is strictly above of surface2, i.e. all its positions
    are strictly smaller than all of surface2's positions.

    Surfaces should have been transposed as needed to make this true.

    Example:
    ┌────────────┐
    │#........###│
    │###......###│
    │############│
    │############│
    │####.......#│
    └────────────┘

    Top set of . characters is the surface1, the bottom set is surface2.
    """

    start = surface1.random_position()
    end = surface2.random_position()

    turn_y = surface1.rng.integers(surface1.max_y, surface2.min_y + 1, dtype=int)

    lines = [
        # Note: off-by-one errors here were quite annoying, be careful.
        Line("vertical", start, turn_y - start[1] + 1),
        Line("horizontal", (start[0], turn_y), end[0] - start[0]),
        Line("vertical", end, turn_y - end[1] - 1),
    ]
    return lines


class BSPTree:
    """
    Split the grid into zones, and return the zones and the index of the first leaf zone.

    This function is used in:
    1) BSP scene that creates rooms at leaf zones and connects them with corridors.
    2) BSPLayout scene that just creates a grid of zones, without rooms or corridors.
    """

    zones: list[Zone]

    def __init__(
        self,
        width: int,
        height: int,
        leaf_zone_count: int,
        rng: np.random.Generator,
    ):
        next_split_id = 0

        # Store the tree as flat list:
        # [
        #   layer1,
        #   layer2, layer2,
        #   layer3, layer3, layer3, layer3,
        #   ...
        # ]
        self.zones = [Zone(0, 0, width, height, rng)]

        for _ in range(leaf_zone_count - 1):  # split rooms-1 times
            zone = self.zones[next_split_id]

            (child1, child2) = zone.split()
            if rng.random() < 0.5:
                child1, child2 = child2, child1

            self.zones.append(child1)
            self.zones.append(child2)
            next_split_id += 1

        self.first_leaf_index = next_split_id

    def get_leaf_zones(self) -> list["Zone"]:
        return self.zones[self.first_leaf_index :]

    def get_all_zones(self) -> list["Zone"]:
        return self.zones

    def get_sibling_pairs(self) -> list[Tuple["Zone", "Zone"]]:
        pairs = []
        for i in range(len(self.zones) - 2, 0, -2):
            pairs.append((self.zones[i], self.zones[i + 1]))
        return pairs
