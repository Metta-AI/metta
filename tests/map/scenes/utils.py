import numpy as np
import pytest

from metta.map.node import Node
from metta.map.types import ChildrenAction, MapGrid
from metta.map.utils.ascii_grid import add_pretty_border, bordered_text_to_lines
from metta.map.utils.storable_map import grid_to_ascii


def render_node(cls: type[Node], params: dict, shape: tuple[int, int], children: list[ChildrenAction] = []):
    grid = np.full(shape, "empty", dtype="<U50")
    node = cls(grid=grid, params=params, children=children)
    node.render_with_children()
    return node


def assert_grid(node: Node, ascii_grid: str):
    grid_lines = grid_to_ascii(node.grid)
    expected_lines, _, _ = bordered_text_to_lines(ascii_grid)

    if grid_lines != expected_lines:
        expected_grid = "\n".join(add_pretty_border(expected_lines))
        actual_grid = "\n".join(add_pretty_border(grid_lines))
        pytest.fail(f"Grid does not match expected:\nEXPECTED:\n{expected_grid}\n\nACTUAL:\n{actual_grid}")


def is_connected(grid: MapGrid):
    """Check if all empty cells in the grid are connected."""
    height, width = grid.shape

    # Find all empty cells
    empty_cells = set()
    for r in range(height):
        for c in range(width):
            if grid[r, c] == "empty":
                empty_cells.add((r, c))

    if not empty_cells:
        return True  # No empty cells means trivially connected

    # Start BFS from any empty cell
    start = next(iter(empty_cells))
    visited = set()
    queue = [start]
    visited.add(start)

    while queue:
        r, c = queue.pop(0)

        # Check all 4 directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc

            # Check bounds and if it's an empty cell we haven't visited
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in visited and grid[nr, nc] == "empty":
                visited.add((nr, nc))
                queue.append((nr, nc))

    # All empty cells are connected if we visited all of them
    return len(visited) == len(empty_cells)


def assert_connected(grid: MapGrid):
    if not is_connected(grid):
        pytest.fail("Grid is not connected:\n" + "\n".join(grid_to_ascii(grid, border=True)))
