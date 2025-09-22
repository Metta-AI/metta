"""Tests for polymorphic serialization of MapBuilderConfig using Pydantic V2 discriminated unions."""

import tempfile
from pathlib import Path

from mettagrid.map_builder import MapBuilderConfig
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.maze import MazePrimMapBuilder
from mettagrid.map_builder.random import RandomMapBuilder
from mettagrid.mapgen.mapgen import MapGen


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

    def test_ascii_config_serialization(self):
        """Test serialization and deserialization of AsciiMapBuilderConfig."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("###\n#.#\n###")
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

    def test_subclass_deserialization(self):
        """Test that subclasses can be deserialized."""
        config = RandomMapBuilder.Config(width=10, height=10, seed=42, objects={"wall": 3})
        json_str = config.model_dump_json()
        deserialized = MapBuilderConfig.model_validate_json(json_str)
        assert isinstance(deserialized, RandomMapBuilder.Config)
        assert deserialized.width == 10
        assert deserialized.height == 10
        assert deserialized.seed == 42
        assert deserialized.objects == {"wall": 3}

    def test_polymorphic_deserialization(self):
        """Test that configs can be serialized to JSON and back."""
        configs = [
            RandomMapBuilder.Config(width=10, height=10, seed=42, objects={"wall": 3}),
            MazePrimMapBuilder.Config(width=15, height=15, start_pos=(1, 1), end_pos=(13, 13), seed=123),
        ]

        for original_config in configs:
            # Serialize to JSON string
            json_str = original_config.model_dump_json()

            # Reconstruct the config
            reconstructed = MapBuilderConfig.model_validate_json(json_str)

            # Compare key attributes
            assert isinstance(reconstructed, original_config.__class__)
            # Same check, but satisfies type checker
            assert isinstance(reconstructed, (RandomMapBuilder.Config, MazePrimMapBuilder.Config))

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

    def test_mapgen(self):
        import mettagrid.mapgen.scenes.random

        map_builder = MapGen.Config(
            num_agents=24,
            width=25,
            height=25,
            border_width=6,
            instance_border_width=0,
            root=mettagrid.mapgen.scenes.random.Random.factory(
                params=mettagrid.mapgen.scenes.random.Random.Params(
                    agents=6,
                    objects={
                        "wall": 10,
                        "altar": 5,
                        "mine_red": 10,
                        "generator_red": 5,
                        "lasery": 1,
                        "armory": 1,
                    },
                ),
            ),
        )
        serialized = map_builder.model_dump()
        assert serialized["type"] == "mettagrid.mapgen.mapgen.MapGen"
