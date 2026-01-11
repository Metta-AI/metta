from typing import Optional

import pytest

from mettagrid.map_builder.utils import create_grid
from mettagrid.mapgen.area import Area
from mettagrid.mapgen.scene import Scene, SceneConfig
from mettagrid.mapgen.types import MapGrid
from mettagrid.mapgen.utils.ascii_grid import add_pretty_border, char_grid_to_lines, default_char_to_name, grid_to_lines


def render_scene(
    scene_cfg: SceneConfig,
    shape: tuple[int, int],
):
    grid = create_grid(shape[0], shape[1])
    area = Area.root_area_from_grid(grid)
    scene = scene_cfg.create_root(area)
    scene.render_with_children()
    return scene


def assert_raw_grid(grid: MapGrid, ascii_grid: str, name_to_char: dict[str, str] | None = None):
    grid_lines = grid_to_lines(grid, name_to_char)
    expected_lines, _, _ = char_grid_to_lines(ascii_grid)

    if grid_lines != expected_lines:
        expected_grid = "\n".join(add_pretty_border(expected_lines))
        actual_grid = "\n".join(add_pretty_border(grid_lines))
        pytest.fail(f"Grid does not match expected:\nEXPECTED:\n{expected_grid}\n\nACTUAL:\n{actual_grid}")


def assert_grid_map(scene: Scene, ascii_grid: str, char_to_name: dict[str, str] | None = None):
    if char_to_name:
        name_to_char: Optional[dict[str, str]] = {}
        # First pass: add all mappings
        for char, name in char_to_name.items():
            if name not in name_to_char:
                name_to_char[name] = char
        # Second pass: prefer visible characters over whitespace
        for char, name in char_to_name.items():
            if char not in (" ", "\t", "\n") and (
                name_to_char.get(name) in (" ", "\t", "\n") or name_to_char[name] == char
            ):
                name_to_char[name] = char
    else:
        name_to_char = None
    assert_raw_grid(scene.grid, ascii_grid, name_to_char)


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


def assert_connected(grid: MapGrid, name_to_char: dict[str, str] | None = None):
    if name_to_char is None:
        # Get default and reverse it
        name_to_char = {v: k for k, v in default_char_to_name().items()}

    if not is_connected(grid):
        pytest.fail("Grid is not connected:\n" + "\n".join(grid_to_lines(grid, name_to_char, border=True)))
