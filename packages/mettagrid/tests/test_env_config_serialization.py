#!/usr/bin/env python3

"""Test environment configuration serialization, particularly polymorphic fields."""

import json

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.map_builder.random import RandomMapBuilder


def test_mg_config_map_builder_serialization():
    """Test that map_builder polymorphic serialization includes all fields."""

    # Create config with specific map_builder parameters
    config = MettaGridConfig.EmptyRoom(num_agents=24, border_width=0)

    # Serialize to JSON and parse back
    config_json = config.model_dump_json(indent=2)
    config_dict = json.loads(config_json)

    # Verify that map_builder contains all expected fields
    map_builder = config_dict["game"]["map_builder"]

    # Check that all RandomMapBuilder.Config fields are present
    assert map_builder["type"] == "mettagrid.map_builder.random.RandomMapBuilder"
    assert "seed" in map_builder
    assert "width" in map_builder
    assert "height" in map_builder
    assert "objects" in map_builder
    assert "agents" in map_builder
    assert "border_width" in map_builder
    assert "border_object" in map_builder

    # Check specific values
    assert map_builder["width"] == 10
    assert map_builder["height"] == 10
    assert map_builder["agents"] == 24
    assert map_builder["border_width"] == 0
    assert map_builder["border_object"] == "wall"


def test_mg_config_custom_map_builder():
    """Test serialization with custom map_builder configuration."""

    # Create custom map builder config
    custom_map_builder = RandomMapBuilder.Config(
        width=15, height=20, agents=12, border_width=2, border_object="stone", objects={"tree": 5, "rock": 3}
    )

    # Create env config with custom map builder
    config = MettaGridConfig()
    config.game.map_builder = custom_map_builder
    config.game.num_agents = 12

    # Serialize and verify
    config_dict = config.model_dump()
    map_builder = config_dict["game"]["map_builder"]

    assert map_builder["type"] == "mettagrid.map_builder.random.RandomMapBuilder"
    assert map_builder["width"] == 15
    assert map_builder["height"] == 20
    assert map_builder["agents"] == 12
    assert map_builder["border_width"] == 2
    assert map_builder["border_object"] == "stone"
    assert map_builder["objects"] == {"tree": 5, "rock": 3}


def test_mg_config_polymorphism_deserialization():
    """Test that we can deserialize the polymorphic map_builder correctly."""

    # Create a config and serialize it
    original_config = MettaGridConfig.EmptyRoom(num_agents=16)
    config_json = original_config.model_dump_json()

    # Deserialize it back
    reconstructed_config = MettaGridConfig.model_validate_json(config_json)

    map_builder_data = reconstructed_config.game.map_builder
    assert isinstance(map_builder_data, RandomMapBuilder.Config)
    assert map_builder_data.agents == 16
    assert map_builder_data.width == 10
    assert map_builder_data.height == 10

    # Test manual reconstruction from the dict data
    map_builder = RandomMapBuilder.Config.model_validate(map_builder_data)
    assert isinstance(map_builder, RandomMapBuilder.Config)
    assert map_builder._builder_cls == RandomMapBuilder
    assert map_builder.agents == 16
    assert map_builder.width == 10
    assert map_builder.height == 10


if __name__ == "__main__":
    test_mg_config_map_builder_serialization()
    test_mg_config_custom_map_builder()
    test_mg_config_polymorphism_deserialization()
    print("All tests passed!")
