from typing import Any
import random
import math
import numpy as np
from mettagrid.map.scene import Scene
from mettagrid.map.node import Node
from mettagrid.map.utils.pattern import Pattern, ascii_to_weights_of_all_patterns


class ConvChain(Scene):
    """
    ConvChain scene generator, based on https://github.com/mxgmn/ConvChain.

    This algorithm generates patterns similar to a given sample pattern.
    It uses a statistical model to capture local features of the sample
    and then generates new patterns with similar local characteristics.

    (Most of this code was written by Claude based on the original C# code.)
    """

    def __init__(
        self,
        pattern: str,
        receptor_size: int = 2,
        iterations: int = 2,
        temperature: float = 0.3,
        children: list[Any] = [],
    ):
        super().__init__(children=children)
        self._pattern = pattern
        self._receptor_size = receptor_size

        self._weights = ascii_to_weights_of_all_patterns(
            self._pattern,
            self._receptor_size,
            # TODO: make these configurable
            periodic=False,
            symmetry="none",
        )
        # Ensure all weights are positive
        self._weights = np.maximum(self._weights, 0.1)

        self._iterations = iterations
        self._temperature = temperature

    def _render(self, node: Node):
        # Generate the field using the ConvChain algorithm
        field = np.random.choice([False, True], size=node.grid.shape)
        n = self._receptor_size
        weights = self._weights

        # Define energy calculation function
        def energy_exp(i: int, j: int) -> float:
            value = 1.0
            for y in range(j - n + 1, j + n):
                for x in range(i - n + 1, i + n):
                    x_wrapped = x % node.width
                    y_wrapped = y % node.height
                    pattern = Pattern(field, x_wrapped, y_wrapped, n)
                    value *= weights[pattern.index()]
            return value

        # Define Metropolis update function
        def metropolis(i: int, j: int) -> None:
            p = energy_exp(i, j)
            field[j, i] = not field[j, i]  # Flip the bit
            q = energy_exp(i, j)

            # Revert the change with some probability
            if math.pow(q / p, 1.0 / self._temperature) < random.random():
                field[j, i] = not field[j, i]  # Flip back

        # Run the Metropolis algorithm
        for _ in range(self._iterations * node.width * node.height):
            i = random.randint(0, node.width - 1)
            j = random.randint(0, node.height - 1)
            metropolis(i, j)

        # Apply the generated field to the node grid
        for y in range(node.height):
            for x in range(node.width):
                node.grid[y, x] = "wall" if field[y, x] else "empty"
