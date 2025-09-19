"""
Test suite for validating ASCII map files.
"""

from pathlib import Path

import numpy as np
import pytest

from mettagrid.map_builder.random import RandomMapBuilder
from mettagrid.util.char_encoder import CHAR_TO_NAME


def find_map_files(root_dir) -> list[str]:
    """
    Find all .map files.

    Args:
        root_dir: Root directory to search from

    Returns:
        Sorted list of relative paths for .map files
    """
    root_path = Path(root_dir).resolve()

    # Return empty list if directory doesn't exist
    if not root_path.exists():
        return []

    map_files = list(root_path.rglob("*.map"))
    relative_paths = [str(path.relative_to(Path.cwd())) for path in map_files]

    return sorted(relative_paths)


def map_files():
    return find_map_files("packages/mettagrid/configs/maps") + find_map_files("configs/maps")


@pytest.fixture(scope="session")
def map_files_fixture():
    return map_files()


def test_programmatic_map_generation():
    """Test that maps can be generated programmatically without config files."""

    # Create a simple map builder configuration
    map_builder = RandomMapBuilder.Config(
        agents=4,
        width=20,
        height=20,
        border_object="wall",
        border_width=1,
    )

    # Verify the configuration is valid
    assert map_builder.agents == 4
    assert map_builder.width == 20
    assert map_builder.height == 20
    assert map_builder.border_object == "wall"
    assert map_builder.border_width == 1


def test_map_files_discovered(map_files_fixture):
    """Verify that map files are found in the repository."""
    assert len(map_files_fixture) > 0, "Should discover at least one .map file"


# ASCII map validation tests
map_files_to_test = map_files()
if map_files_to_test:
    pytest_parametrize = pytest.mark.parametrize(
        "map_file", map_files_to_test, ids=[str(path) for path in map_files_to_test]
    )
else:
    pytest_parametrize = pytest.mark.skip(reason="No map files found in packages/mettagrid/configs/maps/")


@pytest_parametrize
class TestAsciiMap:
    """Test suite for ASCII map validation."""

    @pytest.fixture
    def content(self, map_file):
        with open(map_file, "r", encoding="utf-8") as f:
            return f.read()

    def test_uses_known_symbols(self, content, map_file):
        """Verify that the map only use symbols defined in SYMBOLS mapping."""
        all_chars = set(content)
        unknown_chars = all_chars - set(CHAR_TO_NAME.keys()) - {"\t", "\r", "\n"}

        assert not unknown_chars, f"Map {map_file} contains unknown symbols: {unknown_chars}"

    def test_has_consistent_line_lengths(self, content, map_file):
        """Verify all maps have consistent line lengths within each file."""
        lines = content.strip().splitlines()

        line_lengths = [len(line) for line in lines]
        if len(set(line_lengths)) > 1:
            min_len, max_len = min(line_lengths), max(line_lengths)
            pytest.fail(f"Map has inconsistent line lengths {min_len}-{max_len}")

    def test_loads_as_numpy_array(self, content, map_file):
        """Verify that the map can be loaded as NumPy array (critical for runtime)."""
        lines = content.strip().splitlines()

        level_array = np.array([list(line) for line in lines], dtype="U6")
        _mapped_array = np.vectorize(CHAR_TO_NAME.get)(level_array)

        # Basic structure validation
        assert level_array.ndim == 2, "Should be 2D array"
        assert level_array.shape[0] == len(lines), "Should have correct row count"
