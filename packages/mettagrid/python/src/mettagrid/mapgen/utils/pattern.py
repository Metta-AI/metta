from typing import Literal

import numpy as np
import numpy.typing as npt

from mettagrid.mapgen.utils.ascii_grid import char_grid_to_lines

Symmetry = Literal["all", "horizontal", "none"]


def parse_ascii_into_grid(ascii_source: str) -> npt.NDArray[np.bool_]:
    """
    Parse an ascii string into a numpy ndarray of booleans.

    The string must be composed of `#` and `.` characters.

    `#` will be treated as walls, and `.` as empty.

    Example source:
    #.#
    #.#
    #.#
    """
    lines, width, height = char_grid_to_lines(ascii_source)
    for line in lines:
        if not all(c == "#" or c == "." for c in line):
            raise ValueError("Pattern must be composed of # and . characters")

    grid: npt.NDArray[np.bool_] = np.zeros((height, width), dtype=bool)
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            grid[y, x] = char == "#"

    return grid


def ascii_to_patterns_with_counts(
    ascii_source: str, n: int, periodic: bool, symmetry: Symmetry
) -> list[tuple["Pattern", int]]:
    # This function is useful for WFC - we need to get patterns, not just weights.

    # TODO - support >2 colors?

    grid = parse_ascii_into_grid(ascii_source)

    # pattern index -> { pattern, count }
    seen_patterns = {}

    # Calculate weights from the sample
    max_y = grid.shape[0] if periodic else grid.shape[0] - n + 1
    max_x = grid.shape[1] if periodic else grid.shape[1] - n + 1
    for y in range(max_y):
        for x in range(max_x):
            pattern = Pattern(grid, x, y, n)

            for p in pattern.variations(symmetry):
                if p.index() not in seen_patterns:
                    seen_patterns[p.index()] = {"pattern": p, "count": 0}
                seen_patterns[p.index()]["count"] += 1

    return [(v["pattern"], v["count"]) for v in seen_patterns.values()]


def ascii_to_weights_of_all_patterns(
    source: str, n: int, periodic: bool, symmetry: Symmetry
) -> npt.NDArray[np.float64]:
    # This function is useful for ConvChain. We get weights for all possible patterns, even the ones that
    # don't exist in the sample. (2^(N*N) patterns)

    patterns_with_counts = ascii_to_patterns_with_counts(source, n, periodic, symmetry)

    weights = np.zeros(1 << (n * n))

    for pattern, count in patterns_with_counts:
        index = pattern.index()
        if index >= len(weights):
            raise ValueError(f"Pattern index {index} is out of range for weights array of size {len(weights)}")
        weights[index] = count

    return weights


class Pattern:
    """
    Helper class for handling patterns in the ConvChain and WFC algorithms.

    Currently this class supports only boolean patterns (walls or empty).
    """

    def __init__(self, field: np.ndarray, x: int, y: int, size: int):
        self.data = np.zeros((size, size), dtype=bool)
        field_height, field_width = field.shape

        for j in range(size):
            for i in range(size):
                wrapped_x = (x + i) % field_width
                wrapped_y = (y + j) % field_height
                self.data[j, i] = field[wrapped_y, wrapped_x]

    def size(self) -> int:
        """Return the size of the pattern."""
        return self.data.shape[0]

    def rotated(self) -> "Pattern":
        """Return a new pattern that is this pattern rotated 90 degrees clockwise."""
        result = Pattern.__new__(Pattern)
        size = self.size()
        result.data = np.zeros((size, size), dtype=bool)

        for y in range(size):
            for x in range(size):
                result.data[y, x] = self.data[size - 1 - x, y]

        return result

    def reflected(self) -> "Pattern":
        """Return a new pattern that is this pattern reflected horizontally."""
        result = Pattern.__new__(Pattern)
        size = self.size()
        result.data = np.zeros((size, size), dtype=bool)

        for y in range(size):
            for x in range(size):
                result.data[y, x] = self.data[y, size - 1 - x]

        return result

    def variations(self, symmetry: Symmetry) -> list["Pattern"]:
        if symmetry == "all":
            p0 = self
            p1 = p0.reflected()
            p2 = p0.rotated()
            p3 = p2.reflected()
            p4 = p2.rotated()
            p5 = p4.reflected()
            p6 = p4.rotated()
            p7 = p6.reflected()
            return [p0, p1, p2, p3, p4, p5, p6, p7]
        elif symmetry == "horizontal":
            return [self, self.reflected()]
        elif symmetry == "none":
            return [self]

    def index(self) -> int:
        """Convert the pattern to an integer index for the weights array."""

        result = 0
        size = self.size()

        for y in range(size):
            for x in range(size):
                if self.data[y, x]:
                    result += 1 << (y * size + x)

        return result

    def is_compatible(self, other: "Pattern", dx: int, dy: int) -> bool:
        # """Check if this pattern is compatible with another pattern, given a relative position."""

        size = self.size()
        if size != other.size():
            raise ValueError("Patterns must be the same size")

        # Calculate the overlapping region
        xmin = 0 if dx < 0 else dx
        xmax = size + dx if dx < 0 else size
        ymin = 0 if dy < 0 else dy
        ymax = size + dy if dy < 0 else size

        # Check if patterns agree in the overlapping region
        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                if self.data[y, x] != other.data[y - dy, x - dx]:
                    return False

        return True

    def __str__(self) -> str:
        return "Pattern:\n" + "\n".join(
            "".join("#" if self.data[y, x] else " " for x in range(self.size())) for y in range(self.size())
        )
