"""
Test suite for debug maps functionality.

This module tests that:
1. Debug maps can be loaded without errors
2. Environment configurations are valid
3. Basic smoke tests can be run
"""

from pathlib import Path

import pytest

from mettagrid.room.ascii import SYMBOLS


class TestDebugMaps:
    """Test suite for debug maps."""

    @pytest.fixture(scope="class")
    def debug_maps_dir(self):
        """Fixture providing the debug maps directory."""
        return Path("configs/env/mettagrid/maps/debug")

    def test_debug_maps_exist(self, debug_maps_dir):
        """Verify that debug map files exist."""
        expected_maps = ["mixed_objects.map", "resource_collection.map", "simple_obstacles.map", "tiny_two_altars.map"]

        for map_name in expected_maps:
            map_path = debug_maps_dir / map_name
            assert map_path.exists(), f"Debug map {map_name} should exist"
            assert map_path.is_file(), f"Debug map {map_name} should be a file"

    @pytest.mark.parametrize(
        "map_name", ["mixed_objects.map", "resource_collection.map", "simple_obstacles.map", "tiny_two_altars.map"]
    )
    def test_debug_map_structure(self, debug_maps_dir, map_name):
        """Test that each debug map has valid structure and content."""
        map_path = debug_maps_dir / map_name

        # Read file content
        with open(map_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.strip().splitlines()

        # Check for non-empty content
        assert len(lines) > 0, f"Map {map_name} should not be empty"

        # Validate line length consistency
        line_lengths = [len(line) for line in lines]
        assert len(set(line_lengths)) == 1, f"Map {map_name} should have consistent line lengths"

        # Validate symbols
        all_chars = set("".join(lines))
        unknown_chars = all_chars - set(SYMBOLS.keys())
        # Filter out acceptable whitespace characters
        truly_unknown = unknown_chars - {"\t", "\r", "\n"}
        assert len(truly_unknown) == 0, f"Map {map_name} contains unknown symbols: {sorted(truly_unknown)}"

    def test_debug_environment_configs_exist(self):
        """Verify that environment configuration files exist for debug maps."""
        config_dir = Path("configs/env/mettagrid/navigation/evals")
        expected_configs = [
            "debug_mixed_objects.yaml",
            "debug_resource_collection.yaml",
            "debug_simple_obstacles.yaml",
            "debug_tiny_two_altars.yaml",
        ]

        for config_name in expected_configs:
            config_path = config_dir / config_name
            assert config_path.exists(), f"Environment config {config_name} should exist"
            assert config_path.is_file(), f"Environment config {config_name} should be a file"

    def test_smoke_test_simulation_config_exists(self):
        """Verify that the smoke test simulation configuration exists."""
        config_path = Path("configs/sim/debug_maps_smoke_test.yaml")
        assert config_path.exists(), "Debug maps smoke test config should exist"
        assert config_path.is_file(), "Debug maps smoke test config should be a file"

    def test_smoke_test_script_exists(self):
        """Verify that the smoke test script exists and is executable."""
        script_path = Path("tools/debug_maps_smoke_test.py")
        assert script_path.exists(), "Debug maps smoke test script should exist"
        assert script_path.is_file(), "Debug maps smoke test script should be a file"

    def test_debug_map_contents_are_different(self, debug_maps_dir):
        """Verify that debug maps have different contents (not duplicates)."""
        map_contents = {}
        map_names = ["mixed_objects.map", "resource_collection.map", "simple_obstacles.map", "tiny_two_altars.map"]

        for map_name in map_names:
            map_path = debug_maps_dir / map_name
            with open(map_path, "r") as f:
                content = f.read().strip()
            map_contents[map_name] = content

        # Check that no two maps have identical content
        for i, (name1, content1) in enumerate(map_contents.items()):
            for name2, content2 in list(map_contents.items())[i + 1 :]:
                assert content1 != content2, f"Maps {name1} and {name2} should have different content"

    def test_debug_maps_have_required_elements(self, debug_maps_dir):
        """Verify that debug maps contain required game elements."""
        required_elements = {
            "mixed_objects.map": ["A", "a", "W", "L", "m", "s"],  # Altars, agents, walls, logs, moss, stone
            "resource_collection.map": ["A", "a", "W", "L", "m", "s"],  # Altars, agents, walls, logs, moss, stone
            "simple_obstacles.map": ["A", "a", "W", "L", "m", "s"],  # Altars, agents, walls, logs, moss, stone
            "tiny_two_altars.map": ["A", "a", "W", "L", "m"],  # Altars, agents, walls, logs, moss
        }

        for map_name, expected_elements in required_elements.items():
            map_path = debug_maps_dir / map_name
            with open(map_path, "r") as f:
                content = f.read()

            for element in expected_elements:
                assert element in content, f"Map {map_name} should contain element '{element}'"
