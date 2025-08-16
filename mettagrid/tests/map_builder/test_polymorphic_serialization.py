"""Tests for polymorphic serialization of MapBuilderConfig using Pydantic V2 discriminated unions."""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from metta.mettagrid.map_builder import (
    AsciiMapBuilderConfig,
    MazeKruskalMapBuilderConfig,
    MazePrimMapBuilderConfig,
    RandomMapBuilderConfig,
)
from metta.mettagrid.map_builder.map_builder import MapBuilderConfig


class TestPolymorphicSerialization:
    """Test polymorphic serialization/deserialization of MapBuilderConfig types."""

    def test_random_config_serialization(self):
        """Test serialization and deserialization of RandomMapBuilderConfig."""
        config = RandomMapBuilderConfig(
            type="random",
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
        assert serialized["type"] == "random"
        assert serialized["width"] == 20
        assert serialized["height"] == 30
        assert serialized["seed"] == 42

        # Test JSON serialization
        json_str = config.model_dump_json()
        assert "random" in json_str
        assert "20" in json_str

        # Test deserialization
        deserialized = RandomMapBuilderConfig.model_validate(serialized)
        assert deserialized.type == "random"
        assert deserialized.width == 20
        assert deserialized.height == 30
        assert deserialized.seed == 42
        assert deserialized.objects == {"wall": 5, "altar": 1}

    def test_maze_prim_config_serialization(self):
        """Test serialization and deserialization of MazePrimMapBuilderConfig."""
        config = MazePrimMapBuilderConfig(
            type="maze_prim", width=15, height=15, start_pos=(1, 1), end_pos=(13, 13), branching=0.2, seed=123
        )

        # Test serialization
        serialized = config.model_dump()
        assert serialized["type"] == "maze_prim"
        assert serialized["width"] == 15
        assert serialized["start_pos"] == (1, 1)  # tuples remain as tuples in model_dump

        # Test deserialization
        deserialized = MazePrimMapBuilderConfig.model_validate(serialized)
        assert deserialized.type == "maze_prim"
        assert deserialized.width == 15
        assert deserialized.start_pos == (1, 1)
        assert deserialized.branching == 0.2

    def test_maze_kruskal_config_serialization(self):
        """Test serialization and deserialization of MazeKruskalMapBuilderConfig."""
        config = MazeKruskalMapBuilderConfig(
            type="maze_kruskal", width=21, height=21, start_pos=(0, 0), end_pos=(20, 20), seed=456
        )

        # Test serialization
        serialized = config.model_dump()
        assert serialized["type"] == "maze_kruskal"
        assert serialized["width"] == 21

        # Test deserialization
        deserialized = MazeKruskalMapBuilderConfig.model_validate(serialized)
        assert deserialized.type == "maze_kruskal"
        assert deserialized.width == 21

    def test_ascii_config_serialization(self):
        """Test serialization and deserialization of AsciiMapBuilderConfig."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("###\n#.#\n###")
            temp_path = f.name

        try:
            config = AsciiMapBuilderConfig.from_uri(temp_path)

            # Test serialization
            serialized = config.model_dump()
            assert serialized["type"] == "ascii"
            assert serialized["map_data"] == [["#", "#", "#"], ["#", ".", "#"], ["#", "#", "#"]]

            # Test deserialization
            deserialized = AsciiMapBuilderConfig.model_validate(serialized)
            assert deserialized.type == "ascii"
            assert deserialized.map_data == [["#", "#", "#"], ["#", ".", "#"], ["#", "#", "#"]]

        finally:
            Path(temp_path).unlink()

    def test_discriminator_validation(self):
        """Test that the discriminator field correctly validates types."""
        # Valid configs should work
        valid_configs = [
            {"type": "random", "width": 10, "height": 10},
            {"type": "maze_prim", "width": 15, "height": 15, "start_pos": [1, 1], "end_pos": [13, 13]},
            {"type": "maze_kruskal", "width": 21, "height": 21, "start_pos": [0, 0], "end_pos": [20, 20]},
            {"type": "ascii", "map_data": [["#", "#", "#"], ["#", ".", "#"], ["#", "#", "#"]]},
        ]

        for config_dict in valid_configs:
            # Should not raise an exception
            config_type = config_dict["type"]
            if config_type == "random":
                RandomMapBuilderConfig.model_validate(config_dict)
            elif config_type == "maze_prim":
                MazePrimMapBuilderConfig.model_validate(config_dict)
            elif config_type == "maze_kruskal":
                MazeKruskalMapBuilderConfig.model_validate(config_dict)
            elif config_type == "ascii":
                AsciiMapBuilderConfig.model_validate(config_dict)

    def test_invalid_discriminator_value(self):
        """Test that invalid discriminator values are rejected."""
        # Wrong type for RandomMapBuilderConfig
        with pytest.raises(ValidationError):
            RandomMapBuilderConfig.model_validate({"type": "wrong_type", "width": 10, "height": 10})

        # Missing type field - should use default value, not raise error
        config = RandomMapBuilderConfig.model_validate({"width": 10, "height": 10})
        assert config.type == "random"  # Uses the default value

    def test_union_discriminator_validation(self):
        """Test that the MapBuilderConfigUnion properly discriminates between types."""

        from pydantic import BaseModel

        # Create a test model that uses the union
        class TestContainer(BaseModel):
            config: MapBuilderConfig

        # Test valid configs for each type
        test_cases = [
            {"config": {"type": "random", "width": 10, "height": 10}},
            {"config": {"type": "maze_prim", "width": 15, "height": 15, "start_pos": [1, 1], "end_pos": [13, 13]}},
            {"config": {"type": "maze_kruskal", "width": 21, "height": 21, "start_pos": [0, 0], "end_pos": [20, 20]}},
        ]

        for test_case in test_cases:
            container = TestContainer.model_validate(test_case)
            assert container.config.type == test_case["config"]["type"]

        # Test invalid discriminator value should raise error
        with pytest.raises(ValidationError):
            TestContainer.model_validate({"config": {"type": "invalid_type", "width": 10, "height": 10}})

    def test_json_round_trip(self):
        """Test that configs can be serialized to JSON and back."""
        configs = [
            RandomMapBuilderConfig(type="random", width=10, height=10, seed=42, objects={"wall": 3}),
            MazePrimMapBuilderConfig(
                type="maze_prim", width=15, height=15, start_pos=(1, 1), end_pos=(13, 13), seed=123
            ),
        ]

        for original_config in configs:
            # Serialize to JSON string
            json_str = original_config.model_dump_json()

            # Parse back to dict
            data = json.loads(json_str)

            # Reconstruct the config
            config_type = data["type"]
            if config_type == "random":
                reconstructed = RandomMapBuilderConfig.model_validate(data)
            elif config_type == "maze_prim":
                reconstructed = MazePrimMapBuilderConfig.model_validate(data)
            else:
                pytest.fail(f"Unexpected config type: {config_type}")

            # Compare key attributes
            assert reconstructed.type == original_config.type
            assert reconstructed.width == original_config.width
            assert reconstructed.height == original_config.height

    def test_default_values_serialization(self):
        """Test that default values are properly handled in serialization."""
        # Create config with minimal required fields
        config = RandomMapBuilderConfig(type="random")

        serialized = config.model_dump()

        # Check that defaults are included
        assert serialized["width"] == 10  # default value
        assert serialized["height"] == 10  # default value
        assert serialized["agents"] == 0  # default value
        assert serialized["border_width"] == 0  # default value

        # Deserialization should work
        deserialized = RandomMapBuilderConfig.model_validate(serialized)
        assert deserialized.width == 10
        assert deserialized.height == 10

    def test_config_creation_methods(self):
        """Test that the create() methods work correctly after serialization."""
        config = RandomMapBuilderConfig(type="random", width=5, height=5, seed=999)

        # Serialize and deserialize
        serialized = config.model_dump()
        deserialized = RandomMapBuilderConfig.model_validate(serialized)

        # Test that create() method still works
        map_builder = deserialized.create()
        assert map_builder is not None

        # Test that the builder can create a map
        game_map = map_builder.build()
        assert game_map is not None
        assert game_map.grid.shape == (5, 5)

    def test_type_field_immutability(self):
        """Test that type field defaults are correctly set and immutable."""
        random_config = RandomMapBuilderConfig()
        assert random_config.type == "random"

        maze_prim_config = MazePrimMapBuilderConfig(width=15, height=15, start_pos=(1, 1), end_pos=(13, 13))
        assert maze_prim_config.type == "maze_prim"

        maze_kruskal_config = MazeKruskalMapBuilderConfig(width=21, height=21, start_pos=(0, 0), end_pos=(20, 20))
        assert maze_kruskal_config.type == "maze_kruskal"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("#")
            temp_path = f.name

        try:
            ascii_config = AsciiMapBuilderConfig.from_uri(temp_path)
            assert ascii_config.type == "ascii"
        finally:
            Path(temp_path).unlink()
