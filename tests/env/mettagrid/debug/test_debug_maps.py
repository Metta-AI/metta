"""
Test suite for debug maps functionality.

This module tests that:
1. Debug maps can be loaded without errors
2. Environment configurations are valid
3. Basic smoke tests can be run through the simulation infrastructure
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
        config_dir = Path("configs/env/mettagrid/debug/evals")
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

    def test_debug_defaults_config_exists(self):
        """Verify that the debug defaults configuration exists."""
        config_path = Path("configs/env/mettagrid/debug/evals/defaults.yaml")
        assert config_path.exists(), "Debug defaults config should exist"
        assert config_path.is_file(), "Debug defaults config should be a file"

    def test_smoke_test_simulation_config_exists(self):
        """Verify that the smoke test simulation configuration exists."""
        config_path = Path("configs/sim/debug_maps_smoke_test.yaml")
        assert config_path.exists(), "Debug maps smoke test config should exist"
        assert config_path.is_file(), "Debug maps smoke test config should be a file"

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

    @pytest.mark.parametrize(
        "env_config",
        [
            "env/mettagrid/debug/evals/debug_mixed_objects",
            "env/mettagrid/debug/evals/debug_resource_collection",
            "env/mettagrid/debug/evals/debug_simple_obstacles",
            "env/mettagrid/debug/evals/debug_tiny_two_altars",
        ],
    )
    def test_debug_environment_configs_loadable(self, env_config):
        """Test that debug environment configurations can be loaded by Hydra."""
        import hydra

        try:
            # Use relative path from the test directory to the configs directory
            with hydra.initialize(version_base=None, config_path="../../../../configs"):
                cfg = hydra.compose(config_name=env_config)

                # Access the nested configuration structure
                debug_config = cfg.env.mettagrid.debug.evals

                # Basic validation that config loaded correctly
                assert "game" in debug_config, f"Config {env_config} should have 'game' section"
                assert "map_builder" in debug_config.game, f"Config {env_config} should have map_builder"
                assert "uri" in debug_config.game.map_builder, f"Config {env_config} should have map URI"
                assert "max_steps" in debug_config.game, f"Config {env_config} should have max_steps"

                # Verify it points to correct debug map
                map_uri = debug_config.game.map_builder.uri
                assert "debug" in map_uri, f"Config {env_config} should point to debug map"

        except Exception as e:
            pytest.fail(f"Failed to load environment config {env_config}: {e}")

    def test_smoke_test_simulation_config_loadable(self):
        """Test that the smoke test simulation configuration can be loaded."""
        import hydra

        try:
            with hydra.initialize(version_base=None, config_path="../../../configs"):
                cfg = hydra.compose(config_name="sim/debug_maps_smoke_test")

                # Basic validation
                assert "simulations" in cfg, "Smoke test config should have simulations"
                assert len(cfg.simulations) == 4, "Should have 4 debug map simulations"

                # Check each simulation points to debug environment
                for sim_name, sim_config in cfg.simulations.items():
                    assert "env" in sim_config, f"Simulation {sim_name} should have env config"
                    assert "debug" in sim_config.env, f"Simulation {sim_name} should use debug environment"

        except Exception as e:
            pytest.fail(f"Failed to load smoke test simulation config: {e}")


class TestDebugMapsSmoke:
    """Smoke tests for debug maps that can be run without requiring trained policies."""

    def test_debug_map_loading_smoke_test(self):
        """Smoke test to verify debug maps can be loaded into ASCII room builders."""
        from mettagrid.room.ascii import Ascii

        debug_maps = [
            "configs/env/mettagrid/maps/debug/mixed_objects.map",
            "configs/env/mettagrid/maps/debug/resource_collection.map",
            "configs/env/mettagrid/maps/debug/simple_obstacles.map",
            "configs/env/mettagrid/maps/debug/tiny_two_altars.map",
        ]

        for map_path in debug_maps:
            try:
                # Try to create ASCII room builder with the map
                room_builder = Ascii(uri=map_path, border_width=1)

                # Basic validation that room builder was created successfully
                assert room_builder is not None, f"Room builder should be created for {map_path}"

            except Exception as e:
                pytest.fail(f"Failed to load debug map {map_path} into ASCII room builder: {e}")

    def test_debug_environments_can_be_instantiated(self):
        """Smoke test to verify debug environments can be instantiated with Hydra."""
        import hydra
        from hydra.core.global_hydra import GlobalHydra

        debug_configs = [
            "env/mettagrid/debug/evals/debug_tiny_two_altars"  # Test the smallest map
        ]

        for config_name in debug_configs:
            try:
                # Clear any existing Hydra instance
                GlobalHydra.instance().clear()

                with hydra.initialize(version_base=None, config_path="../../../../configs"):
                    cfg = hydra.compose(config_name=config_name)

                    # Access the nested configuration structure
                    debug_config = cfg.env.mettagrid.debug.evals

                    # Verify we can at least load the configuration structure
                    assert debug_config.game.map_builder._target_ == "mettagrid.room.ascii.Ascii"
                    assert "debug" in debug_config.game.map_builder.uri
                    assert debug_config.game.max_steps > 0

            except Exception as e:
                pytest.fail(f"Failed to instantiate debug environment {config_name}: {e}")
            finally:
                # Clean up Hydra instance
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()
