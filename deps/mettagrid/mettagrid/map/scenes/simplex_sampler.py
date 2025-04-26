from collections import defaultdict
from collections.abc import Callable
from queue import PriorityQueue
from typing import Any, NamedTuple

import hydra
import numpy as np
from opensimplex import OpenSimplex

from mettagrid.map.node import Node
from mettagrid.map.scene import Scene
from mettagrid.map.utils.random import MaybeSeed


# This thing contains all relevant information for each layer:
# exact sampling function, its overall saturation (makes walls and features thicker) and its parameters
# The information itself is written in yaml files. Like this one configs/game/map_builder/mapgen_simsam_basic_demo.yaml
class Layer(NamedTuple):
    sampling_function: Callable
    saturation: float
    function_parameters: dict[str, Any]


class SimplexSampler(Scene):
    EMPTY, WALL, RESOLUTION = "empty", "wall", 255  # RESOLUTION dictates at how many discrete levels terrain is split

    def __init__(
        self,
        seed: MaybeSeed = None,
        children: list[Any] = [],  # it has something to do with Slava's implementation of recursive generation
        # probably parameters for deeper layers, he should know better
        layers: list[Layer] = [],  # layers of sampling functions each of which dictate how generated noise is sampled.
        # All of them are then combined to produce the end result
        cutoff: float = 0.27,  # final step parameter that dictates at which point the whole map will be cut off to
        # walls and empty spaces. Can be between 0 and 1
        force_seed: int = 0,  # Option to use fixed seed.
        # Useful for generating slowly changing variants of one bechmark. 0 will use random seed.
    ):
        super().__init__(children=children)
        self.seed = seed
        self.scaled_cutoff = int(cutoff * self.RESOLUTION)
        self._rng = np.random.default_rng(seed)
        self.layers = [
            Layer(
                hydra.utils.get_method(
                    x.sampling_function
                ),  # yaml files contain simple strings of addresses, here hydra turns them into Callable functions
                x.saturation,
                x.function_parameters,
            )
            for x in layers
            if isinstance(x.sampling_function, str)
        ]
        self.force_seed = force_seed

    def _render(self, node: Node) -> None:
        grid = node.grid
        self._height, self._width = node.grid.shape

        terrain = np.ones(shape=(grid.shape))  # template neutral terrain of ones (no walls) to prevent type errors
        for layer in self.layers:
            if abs(layer.saturation) > 0.00001:
                terrain *= self.terrain_map(layer)  # terrains layered on top of each other via element-wise product
                # (Hadamard product, not matrix multiplication),
                # terrains with ~0 saturation are skipped from calculation
        terrain = self.normalize_array(terrain)  # sets min value as 0, max as 1 via appropriate rescaling
        terrain = np.floor(terrain * self.RESOLUTION).astype("uint8")

        self.fix_map(terrain)
        # final terrain is then split into walls and empty space according to cut off threshold
        room = np.array(np.where(terrain > self.scaled_cutoff, self.EMPTY, self.WALL), dtype="<U50")
        grid[:] = room  # end result is sent to the recursive drawing method from Slava

    def normalize_array(self, room: np.ndarray) -> np.ndarray:
        norm_arr = (room - np.min(room)) / (np.max(room) - np.min(room))
        return norm_arr

    def terrain_map(self, layer: Layer) -> np.ndarray:
        if self.force_seed == 0:
            simplex = OpenSimplex(self._rng.integers(0, 2**31 - 1))
        else:
            simplex = OpenSimplex(self.force_seed)

        # basically the heart of this generation method
        # Could be made faster by parallelizing noise2 calls
        # since every pixel of the end result generates independently from others
        terrain = np.array(
            [
                simplex.noise2(*layer.sampling_function(x, y, self._width, self._height, **layer.function_parameters))
                for y in range(self._height)
                for x in range(self._width)
            ]
        )  # sampling_function dictates where and how fast noise will be sampled for each pixel in a room
        # absolute value is less important than the gradients of this function
        # the faster noise is sampled: sampling_function(x,y) ~ sampling_function(x+1,y) in some region
        # the less changes will be in this region per pixel, noise will look zoomed in
        # the slower noise is sampled: sampling_function(x,y) !=sampling_function(x+1,y) in some region
        # the more changes will be in this region per pixel, noise will look zoomed out
        terrain = (terrain + 1) / 2  # changes range from [-1,1] to [0,1]
        terrain = terrain.reshape((self._height, self._width))
        terrain = (
            terrain**layer.saturation
        )  # saturates pattern with walls. Helpful since base noise is balanced to be 50/50
        # Saturation 0 makes neutral terrain with 0 walls, saturation ~>10 fills ~everything with walls.

        return terrain

    def fix_map(self, terrain: np.ndarray) -> None:
        start = np.where(terrain > self.scaled_cutoff)
        if start[0].size == 0:
            return
        start = (start[0][0], start[1][0])

        def out_of_bound(coordinates):
            y, x = coordinates[0], coordinates[1]
            if x < 0 or y < 0:
                return True
            if x >= self._width or y >= self._height:
                return True
            return False

        dir_list = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # this algorithm tries to fill every square of the map. Non-walls always filled first
        # At some point it starts to climb walls since everything else is filled
        # the terrain dictates which walls are easier to climb
        # As soon as it touches another non-wall cell, it breaks the route that lead to it
        # works really fast and breaks the weakest and closest to 'empty' state parts of terrain first
        # designed to work with 2d np.arrays of ints from 0 to RESOLUTION

        distances = defaultdict(lambda: [float("inf"), tuple()])
        # distances for each cell will tell the minimally found energy to get there (initially infinite)
        # and pointer to the previous position
        distances[start] = [0, start]
        front = PriorityQueue()
        # front will contain cells that are scheduled for exploration with
        # information about energy it took to reach that position, position itself and previous cell that led here
        front.put((0, start, start))

        while not front.empty():
            current_pos_energy, current_pos, previous_pos = front.get()

            for next_dir in dir_list:
                next_pos = (current_pos[0] + next_dir[0], current_pos[1] + next_dir[1])
                if out_of_bound(next_pos):
                    continue
                if terrain[next_pos] <= self.scaled_cutoff:  # the lower the number the closer it is to being a wall:
                    # 0 is the hardest wall, RESOLUTION is completely empty space.
                    # It's a bit counterintuitive, but it fits with global previous step of saturation.
                    # I can rewrite it, but not sure if it's needed
                    next_pos_energy = (
                        current_pos_energy + 1 + (self.scaled_cutoff - terrain[next_pos])
                    )  # since the lowest numbers represent the hardest walls,
                    # we need to add how far they are from cutoff to correctly find
                    # ~easiest way through walls according to terrain

                else:
                    next_pos_energy = current_pos_energy
                    if current_pos_energy > 0 and distances[next_pos][0] == float(
                        "inf"
                    ):  # if we were forced to climb on walls
                        # (current_pos_energy is more than 0 when we take lowest first)
                        # and we are seeing new non-wall part, we break all the walls that got us here
                        next_pos_energy = 0
                        # we break the single wall that we are standing on
                        distances[next_pos] = [next_pos_energy, current_pos]
                        distances[current_pos][0] = 0
                        terrain[current_pos] = self.RESOLUTION
                        front.put((next_pos_energy, next_pos, current_pos))
                        front.put((next_pos_energy, current_pos, previous_pos))

                        # then we see if there were more in previous chain of steps and break them as well
                        while distances[previous_pos][0] != 0:
                            another = distances[previous_pos][1]
                            distances[previous_pos][0] = 0
                            front.put((0, previous_pos, distances[previous_pos][1]))
                            terrain[previous_pos] = self.RESOLUTION
                            previous_pos = another

                if next_pos_energy >= distances[next_pos][0]:  # type: ignore
                    continue

                distances[next_pos] = [next_pos_energy, current_pos]
                front.put((next_pos_energy, next_pos, current_pos))
