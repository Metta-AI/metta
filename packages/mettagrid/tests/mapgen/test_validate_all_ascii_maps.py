"""
Test suite for validating ASCII map files.
"""

from pathlib import Path

import numpy as np
import pytest

from mettagrid.map_builder.random import RandomMapBuilder
from mettagrid.mapgen.utils.ascii_grid import DEFAULT_CHAR_TO_NAME


def find_map_files(root_dir) -> list[Path]:
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
        # Read map file and, if it is a YAML-based map with a map_data block,
        # extract only the ASCII map content so validations operate on the grid
        # rather than YAML keys. Fallback to the full content for plain ASCII maps.
        with open(str(map_file), "r", encoding="utf-8") as f:
            raw = f.read()

        lines = raw.splitlines()
        map_text_lines: list[str] = []

        # Detect a YAML block scalar for map_data and extract it if present.
        # We avoid adding a dependency on PyYAML by using indentation heuristics.
        map_idx = None
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("map_data:"):
                map_idx = i
                break

        if map_idx is not None and map_idx + 1 < len(lines):
            # The line after map_data: should begin the indented block content.
            # Compute the indentation of the first content line and gather all
            # subsequent lines that are indented at least that much.
            first_content = lines[map_idx + 1]
            indent_width = len(first_content) - len(first_content.lstrip(" "))

            # If indent_width is zero, treat as plain content fallback.
            if indent_width > 0:
                for j in range(map_idx + 1, len(lines)):
                    line_j = lines[j]
                    # Count leading spaces
                    lead = len(line_j) - len(line_j.lstrip(" "))
                    if lead < indent_width:
                        break
                    # Strip only the block indent, keep internal spacing
                    map_text_lines.append(line_j[indent_width:])

        # Fallback to raw if we did not find a map_data block
        if not map_text_lines:
            return raw

        return "\n".join(map_text_lines)

    @pytest.fixture(scope="class")
    def char_to_name(self):
        """Build char_to_name mapping that includes all map characters."""
        from mettagrid.config.mettagrid_config import (
            AssemblerConfig,
            ChestConfig,
            ConverterConfig,
            WallConfig,
        )

        # Create a comprehensive mapping that includes all object types used in maps
        objects = {
            "wall": WallConfig(name="wall", type_id=1, map_char="#", render_symbol="⬛"),
            "converter": ConverterConfig(name="converter", type_id=2, map_char="c", render_symbol="🔄", cooldown=[0]),
            "assembler": AssemblerConfig(name="assembler", type_id=3, map_char="m", render_symbol="🏭"),
            "chest": ChestConfig(
                name="chest",
                type_id=4,
                map_char="n",
                render_symbol="📦",
                resource_type="ore_red",
            ),
            # Common navigation map characters
            "floor": WallConfig(name="floor", type_id=5, map_char="_", render_symbol="░"),
            # Object use map characters
            "swappable_wall": WallConfig(
                name="swappable_wall", type_id=6, map_char="s", render_symbol="▒", swappable=True
            ),
            "special": WallConfig(name="special", type_id=7, map_char="S", render_symbol="✦"),
        }
        return DEFAULT_CHAR_TO_NAME | {o.map_char: o.name for o in objects.values()}

    def test_uses_known_symbols(self, content, map_file, char_to_name):
        """Verify that the map only use symbols defined in config mapping."""
        all_chars = set(content)
        unknown_chars = all_chars - set(char_to_name.keys()) - {"\t", "\r", "\n"}

        assert not unknown_chars, f"Map {map_file} contains unknown symbols: {unknown_chars}"

    def test_has_consistent_line_lengths(self, content, map_file):
        """Verify all maps have consistent line lengths within each file."""
        lines = content.strip().splitlines()

        line_lengths = [len(line) for line in lines]
        if len(set(line_lengths)) > 1:
            min_len, max_len = min(line_lengths), max(line_lengths)
            pytest.fail(f"Map has inconsistent line lengths {min_len}-{max_len}")

    def test_loads_as_numpy_array(self, content, map_file, char_to_name):
        """Verify that the map can be loaded as NumPy array (critical for runtime)."""
        lines = content.strip().splitlines()

        level_array = np.array([list(line) for line in lines], dtype="U6")
        _mapped_array = np.vectorize(char_to_name.get)(level_array)

        # Basic structure validation
        assert level_array.ndim == 2, "Should be 2D array"
        assert level_array.shape[0] == len(lines), "Should have correct row count"
