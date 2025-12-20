"""Tests for polymorphic serialization of MapBuilderConfig using Pydantic V2 discriminated unions."""

import json
import tempfile
import textwrap
from pathlib import Path

import pytest
from pytest import fixture

from mettagrid.base_config import Config
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.map_builder import AnyMapBuilderConfig, MapBuilderConfig
from mettagrid.map_builder.maze import MazePrimMapBuilder
from mettagrid.map_builder.random_map import RandomMapBuilder


def test_random_config_serialization():
    """Test serialization and deserialization of RandomMapBuilderConfig."""
    config = RandomMapBuilder.Config(
        width=20,
        height=30,
        seed=42,
        objects={"wall": 5, "assembler": 1},
        agents=2,
        border_width=1,
        border_object="wall",
    )

    # Test serialization
    serialized = config.model_dump()
    assert serialized["type"] == "mettagrid.map_builder.random_map.RandomMapBuilder.Config"
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
    assert deserialized.objects == {"wall": 5, "assembler": 1}


def test_default_values_serialization():
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


class OuterConfig(Config):
    map_builder: AnyMapBuilderConfig


@fixture
def wrapped_any_config():
    return OuterConfig(
        map_builder=RandomMapBuilder.Config(
            agents=2,
        )
    )


def test_any_config_serialization(wrapped_any_config: OuterConfig):
    serialized = wrapped_any_config.model_dump()
    assert serialized["map_builder"]["type"] == "mettagrid.map_builder.random_map.RandomMapBuilder.Config"
    assert serialized["map_builder"]["agents"] == 2


def test_any_config_deserialization(wrapped_any_config: OuterConfig):
    restored_config = OuterConfig.model_validate(wrapped_any_config.model_dump())

    assert isinstance(restored_config.map_builder, RandomMapBuilder.Config)
    assert restored_config.map_builder.agents == 2


def test_ascii_config_serialization():
    """Test serialization and deserialization of AsciiMapBuilderConfig."""
    yaml_content = """\
type: mettagrid.map_builder.ascii.AsciiMapBuilder.Config
map_data: |-
    ###
    #.#
    ###
char_to_map_name:
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
        assert serialized["type"] == "mettagrid.map_builder.ascii.AsciiMapBuilder.Config"
        assert serialized["map_data"] == [["#", "#", "#"], ["#", ".", "#"], ["#", "#", "#"]]

        # Test deserialization
        deserialized = AsciiMapBuilder.Config.model_validate(serialized)
        assert isinstance(deserialized, AsciiMapBuilder.Config)
        assert deserialized.map_data == [["#", "#", "#"], ["#", ".", "#"], ["#", "#", "#"]]

    finally:
        Path(temp_path).unlink()


@fixture
def random_config_yaml():
    return textwrap.dedent(
        """
        type: mettagrid.map_builder.random_map.RandomMapBuilder.Config
        width: 4
        height: 3
        seed: 99
        objects:
            wall: 2
        agents: 1
        border_width: 1
        border_object: wall
        """
    )


@fixture
def random_config_yaml_path(random_config_yaml: str, tmp_path: Path):
    path = tmp_path / "random_config.yaml"
    with open(path, "w", encoding="utf-8") as tmp:
        tmp.write(random_config_yaml)

    return path


def test_random_config_from_uri(random_config_yaml_path: Path):
    """Random builder config loads via MapBuilderConfig.from_uri."""

    config = MapBuilderConfig.from_uri(random_config_yaml_path)
    assert isinstance(config, RandomMapBuilder.Config)
    assert config.width == 4
    assert config.height == 3
    assert config.seed == 99

    builder = config.create()
    assert isinstance(builder, RandomMapBuilder)

    game_map = builder.build()
    assert game_map.grid.shape == (3, 4)


def test_random_config_from_str(random_config_yaml: str):
    """Random builder config loads via MapBuilderConfig.from_str."""

    config = MapBuilderConfig.from_str(random_config_yaml)
    assert isinstance(config, RandomMapBuilder.Config)
    assert config.width == 4
    assert config.height == 3
    assert config.objects == {"wall": 2}
    assert config.agents == 1

    builder = config.create()
    game_map = builder.build()
    assert game_map.grid.shape == (3, 4)


def test_config_from_str_wrong_class(random_config_yaml: str):
    with pytest.raises(
        TypeError,
        match="RandomMapBuilder.Config is not a subclass of mettagrid.map_builder.ascii.AsciiMapBuilder.Config",
    ):
        AsciiMapBuilder.Config.from_str(random_config_yaml)


def test_json_round_trip():
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
        if config_type == "mettagrid.map_builder.random_map.RandomMapBuilder.Config":
            reconstructed = RandomMapBuilder.Config.model_validate(data)
        elif config_type == "mettagrid.map_builder.maze.MazePrimMapBuilder.Config":
            reconstructed = MazePrimMapBuilder.Config.model_validate(data)
        else:
            pytest.fail(f"Unexpected config type: {config_type}")

        # Compare key attributes
        assert isinstance(reconstructed, original_config.__class__)
        assert reconstructed.width == original_config.width
        assert reconstructed.height == original_config.height


def test_config_creation_methods():
    """Test that the create() methods work correctly after serialization."""
    config = RandomMapBuilder.Config(width=5, height=5, seed=999)

    # Serialize and deserialize
    serialized = config.model_dump()
    deserialized = RandomMapBuilder.Config.model_validate(serialized)

    # Test that create() method still works
    map_builder = deserialized.create()
    assert map_builder is not None
    assert isinstance(map_builder, RandomMapBuilder)

    # Test that the builder can create a map
    game_map = map_builder.build()
    assert game_map is not None
    assert game_map.grid.shape == (5, 5)
