# This code is a Python translation of https://github.com/mxgmn/ConvChain, with some modifications.
#
# Original work Copyright (c) mxgmn 2016
#
# Licensed under the MIT License:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# The software is provided "as is", without warranty of any kind, express or
# implied, including but not limited to the warranties of merchantability,
# fitness for a particular purpose and noninfringement. In no event shall the
# authors or copyright holders be liable for any claim, damages or other
# liability, whether in an action of contract, tort or otherwise, arising from,
# out of or in connection with the software or the use or other dealings in the
# software.

import math

import numpy as np

from mettagrid.base_config import Config
from mettagrid.mapgen.scene import Scene, SceneConfig
from mettagrid.mapgen.utils.pattern import Pattern, Symmetry, ascii_to_weights_of_all_patterns


class BaseConvChainConfig(Config):
    pattern: str
    pattern_size: int
    iterations: int
    temperature: float
    periodic_input: bool = True
    symmetry: Symmetry = "all"


class ConvChainConfig(BaseConvChainConfig, SceneConfig):
    pass


class ConvChain(Scene[ConvChainConfig]):
    """
    ConvChain scene generator, based on https://github.com/mxgmn/ConvChain
    (ConvChainFast.cs version).

    This algorithm generates patterns similar to a given sample pattern.
    It uses a statistical model to capture local features of the sample
    and then generates new patterns with similar local characteristics.
    """

    def post_init(self):
        self._weights = ascii_to_weights_of_all_patterns(
            self.config.pattern,
            self.config.pattern_size,
            periodic=self.config.periodic_input,
            symmetry=self.config.symmetry,
        )
        # Ensure all weights are positive
        self._weights = np.maximum(self._weights, 0.1)

    def render(self):
        config = self.config
        # Intentionally use a plain Python list: this loop is scalar and numpy adds overhead.
        field = self.rng.choice([False, True], size=self.grid.shape).tolist()
        # Keep a local list to reduce attribute lookups inside the hot loop.
        weights = list(self._weights)
        n = config.pattern_size
        # Precompute 2^(dy*n+dx) powers to avoid recomputing inside the inner loop.
        power_lookup = [1 << i for i in range(n * n)]
        width = self.width
        height = self.height

        for _ in range(config.iterations * self.width * self.height):
            x0 = self.rng.integers(0, width, dtype=int)
            y0 = self.rng.integers(0, height, dtype=int)

            # This algorithm applies the same bitwise energy calc as ConvChainFast.cs.
            # For a clearer walkthrough, see ConvChainSlow, or the original C#:
            # https://github.com/mxgmn/ConvChain/blob/master/ConvChainFast.cs
            q = 1
            for sy in range(y0 - n + 1, y0 + n):
                y_vals = [(sy + dy) % height for dy in range(n)]
                for sx in range(x0 - n + 1, x0 + n):
                    x_vals = [(sx + dx) % width for dx in range(n)]
                    ind = 0
                    difference = 0
                    for dy in range(n):
                        for dx in range(n):
                            x = x_vals[dx]
                            y = y_vals[dy]

                            value = field[y][x]
                            power = power_lookup[dy * n + dx]
                            if value:
                                ind += power
                            if x == x0 and y == y0:
                                difference = power if value else -power

                    q *= weights[ind - difference] / weights[ind]

            # For the sake of parity with ConvChainSlow class, we pre-generate a random number.
            # (This allows us to compare whether the output is identical to the ConvChainSlow version;
            # can be optimized later.)
            rnd = self.rng.random()
            if q >= 1:
                field[y0][x0] = not field[y0][x0]
                continue

            if config.temperature != 1:
                q = q ** (1.0 / config.temperature)

            if q > rnd:
                field[y0][x0] = not field[y0][x0]

        # Apply the generated field to the scene grid
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y, x] = "wall" if field[y][x] else "empty"


class ConvChainSlowConfig(BaseConvChainConfig, SceneConfig):
    pass


class ConvChainSlow(Scene[ConvChainSlowConfig]):
    """
    ConvChain scene generator, naive & slow implementation.

    Committed to the repo for the sake of comparison, usually shouldn't be used and can be removed later.
    """

    def post_init(self):
        self._weights = ascii_to_weights_of_all_patterns(
            self.config.pattern,
            self.config.pattern_size,
            periodic=self.config.periodic_input,
            symmetry=self.config.symmetry,
        )
        # Ensure all weights are positive
        self._weights = np.maximum(self._weights, 0.1)

    def render(self):
        # Generate the field using the ConvChain algorithm
        field = self.rng.choice([False, True], size=self.grid.shape)
        n = self.config.pattern_size
        weights = self._weights

        # Define energy calculation function
        def energy_exp(i: int, j: int) -> float:
            value = 1.0
            for y in range(j - n + 1, j + n):
                for x in range(i - n + 1, i + n):
                    x_wrapped = x % self.width
                    y_wrapped = y % self.height
                    pattern = Pattern(field, x_wrapped, y_wrapped, n)
                    value *= weights[pattern.index()]
            return value

        # Define Metropolis update function
        def metropolis(x: int, y: int) -> None:
            p = energy_exp(x, y)
            field[y, x] = not field[y, x]  # Flip the bit
            q = energy_exp(x, y)

            # Revert the change with some probability
            if math.pow(q / p, 1.0 / self.config.temperature) < self.rng.random():
                field[y, x] = not field[y, x]  # Flip back

        # Run the Metropolis algorithm
        for _ in range(self.config.iterations * self.width * self.height):
            x = self.rng.integers(0, self.width, dtype=int)
            y = self.rng.integers(0, self.height, dtype=int)
            metropolis(x, y)

        # Apply the generated field to the scene grid
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y, x] = "wall" if field[y, x] else "empty"
