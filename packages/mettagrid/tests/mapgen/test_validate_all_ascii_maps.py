"""
Test suite for validating ASCII map files.
"""

from pathlib import Path
from typing import Iterable
import sys

import numpy as np
import pytest

from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.random import RandomMapBuilder

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.map.convert_legacy_maps_to_yaml import DEFAULT_DIRECTORIES


def find_map_files(root_dir: Path | str) -> list[Path]:
    """
    Find all .map files.

    Args:
        root_dir: Root directory to search from

    Returns:
        Sorted list of absolute paths for .map files
    """
    root_path = Path(root_dir).resolve()

    # Return empty list if directory doesn't exist
    if not root_path.exists():
        return []

    map_files = list(root_path.rglob("*.map"))

    return sorted(map_files)


def map_files() -> list[Path]:
    candidates: Iterable[Path] = (
        map_file for directory in DEFAULT_DIRECTORIES for map_file in find_map_files(directory)
    )
    return sorted(candidates)


@pytest.fixture(scope="session")
def map_files_fixture() -> list[Path]:
    return map_files()


def test_programmatic_map_generation() -> None:
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


def test_map_files_discovered(map_files_fixture: list[Path]) -> None:
    """Verify that map files are found in the repository."""
    assert len(map_files_fixture) > 0, "Should discover at least one .map file"


# ASCII map validation tests
map_files_to_test = map_files()
if map_files_to_test:
    pytest_parametrize = pytest.mark.parametrize(
        "map_file", map_files_to_test, ids=[str(path) for path in map_files_to_test]
    )
else:
    directories = ", ".join(str(directory) for directory in DEFAULT_DIRECTORIES)
    pytest_parametrize = pytest.mark.skip(reason=f"No map files found in {directories}")


@pytest_parametrize
class TestAsciiMap:
    """Test suite for ASCII map validation."""

    @pytest.fixture
    def ascii_config(self, map_file: Path) -> AsciiMapBuilder.Config:
        return AsciiMapBuilder.Config.from_uri(map_file)

    @pytest.fixture
    def map_rows(self, ascii_config: AsciiMapBuilder.Config) -> list[list[str]]:
        return ascii_config.map_data

    @pytest.fixture
    def char_to_name(self, ascii_config: AsciiMapBuilder.Config) -> dict[str, str]:
        return ascii_config.char_to_name_map

    def test_uses_known_symbols(
        self,
        map_rows: list[list[str]],
        map_file: Path,
        char_to_name: dict[str, str],
    ) -> None:
        """Verify that the map only use symbols defined in the YAML mapping."""
        map_chars = {cell for row in map_rows for cell in row}
        unknown_chars = map_chars - set(char_to_name.keys())

        assert not unknown_chars, f"Map {map_file} contains unknown symbols: {unknown_chars}"

    def test_has_consistent_line_lengths(self, map_rows: list[list[str]], map_file: Path) -> None:
        """Verify all maps have consistent line lengths within each file."""
        line_lengths = [len(row) for row in map_rows]
        if len(set(line_lengths)) > 1:
            min_len, max_len = min(line_lengths), max(line_lengths)
            pytest.fail(f"Map {map_file} has inconsistent line lengths {min_len}-{max_len}")

    def test_loads_as_numpy_array(
        self,
        map_rows: list[list[str]],
        map_file: Path,
        char_to_name: dict[str, str],
    ) -> None:
        """Verify that the map can be loaded as NumPy array (critical for runtime)."""
        level_array = np.array(map_rows, dtype="U6")
        _mapped_array = np.vectorize(char_to_name.get)(level_array)

        assert level_array.ndim == 2, "Should be 2D array"
        assert level_array.shape[0] == len(map_rows), "Should have correct row count"
