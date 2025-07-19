"""
Test suite for validating ASCII map files.
"""

from pathlib import Path

import pytest

from metta.map.utils.ascii_grid import char_grid_to_lines, lines_to_grid, load_map_file, validate_map_file
from metta.mettagrid.char_encoder import CHAR_TO_NAME


def find_map_files(root_dir="configs") -> list[str]:
    """
    Find all .map files.

    Args:
        root_dir: Root directory to search from

    Returns:
        Sorted list of relative paths for .map files
    """
    root_path = Path(root_dir).resolve()
    map_files = list(root_path.rglob("*.map"))
    relative_paths = [str(path.relative_to(Path.cwd())) for path in map_files]
    return sorted(relative_paths)


@pytest.fixture(scope="session")
def map_files():
    return find_map_files()


def test_map_files_discovered(map_files):
    """Verify that map files are found in the repository."""
    assert len(map_files) > 0, "Should discover at least one .map file"


@pytest.mark.parametrize("map_file", find_map_files(), ids=[str(path) for path in find_map_files()])
class TestAsciiMap:
    """Test suite for ASCII map validation."""

    def test_validates_successfully(self, map_file):
        """Verify that the map file passes validation."""
        is_valid, error_msg = validate_map_file(map_file)
        assert is_valid, f"Map {map_file} validation failed: {error_msg}"

    def test_loads_successfully(self, map_file):
        """Verify that the map file can be loaded."""
        lines = load_map_file(map_file)
        assert len(lines) > 0, f"Map {map_file} is empty"

    def test_converts_to_grid(self, map_file):
        """Verify that the map can be converted to MapGrid."""
        lines = load_map_file(map_file)
        grid = lines_to_grid(lines)

        # Basic structure validation
        assert grid.ndim == 2, "Should be 2D array"
        assert grid.shape[0] == len(lines), "Should have correct row count"
        assert grid.shape[1] == len(lines[0]), "Should have correct column count"

        # Verify all cells have valid grid objects
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                assert grid[r, c] in CHAR_TO_NAME.values(), f"Invalid grid object at ({r}, {c}): {grid[r, c]}"

    def test_char_grid_parsing(self, map_file):
        """Verify char_grid_to_lines works correctly."""
        with open(map_file, "r", encoding="utf-8") as f:
            content = f.read()

        if content.strip():  # Only test non-empty files
            lines, width, height = char_grid_to_lines(content)
            assert len(lines) == height, "Height mismatch"
            assert all(len(line) == width for line in lines), "Width inconsistency"
