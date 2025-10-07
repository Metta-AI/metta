"""Tests for polymorphic serialization of MapBuilderConfig using Pydantic V2 discriminated unions."""

import json
import tempfile
from pathlib import Path

import pytest

from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.maze import MazeKruskalMapBuilder, MazePrimMapBuilder
from mettagrid.map_builder.random import RandomMapBuilder


class TestPolymorphicSerialization:
    """Test polymorphic serialization/deserialization of MapBuilderConfig types."""

    def test_random_config_serialization(self):
        """Test serialization and deserialization of RandomMapBuilderConfig."""
        config = RandomMapBuilder.Config(
            width=20,
            height=30,
            seed=42,
            objects={"wall": 5, "altar": 1},
            agents=2,
            border_width=1,
            border_object="wall",
        )

        # Test serialization
        serialized = config.model_dump()
        assert serialized["type"] == "mettagrid.map_builder.random.RandomMapBuilder"
        assert serialized["width"] == 20
        assert serialized["height"] == 30
        assert serialized["seed"] == 42

        # Test JSON serialization
        json_str = config.model_dump_json()
        assert "random" in json_str
        assert "20" in json_str

        # Test deserialization
        deserialized = RandomMapBuilder.Config.model_validate(serialized)
        assert isinstance(deserialized, RandomMapBuilder.Config)
        assert deserialized.width == 20
        assert deserialized.height == 30
        assert deserialized.seed == 42
        assert deserialized.objects == {"wall": 5, "altar": 1}

    def test_maze_prim_config_serialization(self):
        """Test serialization and deserialization of MazePrimMapBuilderConfig."""
        config = MazePrimMapBuilder.Config(
            width=15, height=15, start_pos=(1, 1), end_pos=(13, 13), branching=0.2, seed=123
        )

        # Test serialization
        serialized = config.model_dump()
        assert serialized["type"] == "mettagrid.map_builder.maze.MazePrimMapBuilder"
        assert serialized["width"] == 15
        assert serialized["start_pos"] == (1, 1)  # tuples remain as tuples in model_dump

        # Test deserialization
        deserialized = MazePrimMapBuilder.Config.model_validate(serialized)
        assert isinstance(deserialized, MazePrimMapBuilder.Config)
        assert deserialized.width == 15
        assert deserialized.start_pos == (1, 1)
        assert deserialized.branching == 0.2

    def test_maze_kruskal_config_serialization(self):
        """Test serialization and deserialization of MazeKruskalMapBuilderConfig."""
        config = MazeKruskalMapBuilder.Config(width=21, height=21, start_pos=(0, 0), end_pos=(20, 20), seed=456)

        # Test serialization
        serialized = config.model_dump()
        assert serialized["type"] == "mettagrid.map_builder.maze.MazeKruskalMapBuilder"
        assert serialized["width"] == 21

        # Test deserialization
        deserialized = MazeKruskalMapBuilder.Config.model_validate(serialized)
        assert isinstance(deserialized, MazeKruskalMapBuilder.Config)
        assert deserialized.width == 21

    def test_ascii_config_serialization(self):
        """Test serialization and deserialization of AsciiMapBuilderConfig."""
        yaml_content = """\
map_data: |-
  ###
  #.#
  ###
char_to_name_map:
  "#": wall
  ".": empty
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".map", delete=False, encoding="utf-8") as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = AsciiMapBuilder.Config.from_uri(temp_path)

            # Test serialization
            serialized = config.model_dump()
            assert serialized["type"] == "mettagrid.map_builder.ascii.AsciiMapBuilder"
            assert serialized["map_data"] == [["#", "#", "#"], ["#", ".", "#"], ["#", "#", "#"]]

            # Test deserialization
            deserialized = AsciiMapBuilder.Config.model_validate(serialized)
            assert isinstance(deserialized, AsciiMapBuilder.Config)
            assert deserialized.map_data == [["#", "#", "#"], ["#", ".", "#"], ["#", "#", "#"]]

        finally:
            Path(temp_path).unlink()

    def test_json_round_trip(self):
        """Test that configs can be serialized to JSON and back."""
        configs = [
            RandomMapBuilder.Config(width=10, height=10, seed=42, objects={"wall": 3}),
            MazePrimMapBuilder.Config(width=15, height=15, start_pos=(1, 1), end_pos=(13, 13), seed=123),
        ]

        for original_config in configs:
            # Serialize to JSON string
            json_str = original_config.model_dump_json()

            # Parse back to dict
            data = json.loads(json_str)

            # Reconstruct the config
            config_type = data["type"]
            if config_type == "mettagrid.map_builder.random.RandomMapBuilder":
                reconstructed = RandomMapBuilder.Config.model_validate(data)
            elif config_type == "mettagrid.map_builder.maze.MazePrimMapBuilder":
                reconstructed = MazePrimMapBuilder.Config.model_validate(data)
            else:
                pytest.fail(f"Unexpected config type: {config_type}")

            # Compare key attributes
            assert isinstance(reconstructed, original_config.__class__)
            assert reconstructed.width == original_config.width
            assert reconstructed.height == original_config.height

    def test_default_values_serialization(self):
        """Test that default values are properly handled in serialization."""
        # Create config with minimal required fields
        config = RandomMapBuilder.Config()

        serialized = config.model_dump()

        # Check that defaults are included
        assert serialized["width"] == 10  # default value
        assert serialized["height"] == 10  # default value
        assert serialized["agents"] == 0  # default value
        assert serialized["border_width"] == 0  # default value

        # Deserialization should work
        deserialized = RandomMapBuilder.Config.model_validate(serialized)
        assert deserialized.width == 10
        assert deserialized.height == 10

    def test_config_creation_methods(self):
        """Test that the create() methods work correctly after serialization."""
        config = RandomMapBuilder.Config(width=5, height=5, seed=999)

        # Serialize and deserialize
        serialized = config.model_dump()
        deserialized = RandomMapBuilder.Config.model_validate(serialized)

        # Test that create() method still works
        map_builder = deserialized.create()
        assert map_builder is not None

        # Test that the builder can create a map
        game_map = map_builder.build()
        assert game_map is not None
        assert game_map.grid.shape == (5, 5)
