import numpy as np
import pytest

from metta.map.random.int import MaybeSeed
from metta.map.scene import Scene
from metta.map.types import Area, ChildrenAction, MapGrid
from metta.map.utils.ascii_grid import add_pretty_border, char_grid_to_lines
from metta.map.utils.storable_map import grid_to_lines


def render_scene(
    cls: type[Scene],
    params: dict,
    shape: tuple[int, int],
    children: list[ChildrenAction] | None = None,
    seed: MaybeSeed = None,
):
    grid = np.full(shape, "empty", dtype="<U50")
    area = Area.root_area_from_grid(grid)
    scene = cls(area=area, params=params, children=children or [], seed=seed)
    scene.render_with_children()
    return scene


def assert_grid(scene: Scene, ascii_grid: str):
    grid_lines = grid_to_lines(scene.grid)
    expected_lines, _, _ = char_grid_to_lines(ascii_grid)

    if grid_lines != expected_lines:
        expected_grid = "\n".join(add_pretty_border(expected_lines))
        actual_grid = "\n".join(add_pretty_border(grid_lines))
        pytest.fail(f"Grid does not match expected:\nEXPECTED:\n{expected_grid}\n\nACTUAL:\n{actual_grid}")


def is_connected(grid: MapGrid):
    """Check if all empty cells in the grid are connected."""
    height, width = grid.shape

    def is_empty(cell) -> bool:
        return cell == "empty" or cell.startswith("agent")

    # Find all empty cells
    empty_cells = set()
    for r in range(height):
        for c in range(width):
            if is_empty(grid[r, c]):
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
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in visited and is_empty(str(grid[nr, nc])):
                visited.add((nr, nc))
                queue.append((nr, nc))

    # All empty cells are connected if we visited all of them
    return len(visited) == len(empty_cells)


def assert_connected(grid: MapGrid):
    if not is_connected(grid):
        pytest.fail("Grid is not connected:\n" + "\n".join(grid_to_lines(grid, border=True)))
