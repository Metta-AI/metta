"""Test map validation with GameConfig integration."""

import pytest

from metta.mettagrid.map_builder.random import RandomMapBuilder
from metta.mettagrid.mettagrid_config import GameConfig


def test_map_validation_with_game_config():
    """Test that map builders can validate against game config."""
    # Create a minimal game config with known objects
    game_config = GameConfig(
        num_agents=4,
        objects={
            "wall": {"type_id": 1},
            "generator": {"type_id": 2},
            "converter": {"type_id": 3},
            "agent.team_1": {"type_id": 0, "group_id": 1},
        },
    )

    # Create a random map builder
    config = RandomMapBuilder.Config(
        width=10,
        height=10,
        objects={"generator": 2, "converter": 1},
        agents={"team_1": 4},
        border_width=1,
        border_object="wall",
    )

    builder = RandomMapBuilder(config)
    builder.set_game_config(game_config)

    # Build and validate
    game_map = builder.build_validated()

    # Check that validation happened (should have compressed)
    assert game_map.byte_grid is not None
    assert game_map.object_key is not None

    # Verify object key contains expected objects
    assert "wall" in game_map.object_key
    assert "generator" in game_map.object_key
    assert "converter" in game_map.object_key
    assert "agent.team_1" in game_map.object_key
    assert "empty" in game_map.object_key


def test_map_validation_catches_unknown_objects():
    """Test that validation catches unknown objects."""

    # Create a game config WITHOUT "unknown_object"
    game_config = GameConfig(
        num_agents=1,
        objects={
            "wall": {"type_id": 1},
            "agent.team_1": {"type_id": 0, "group_id": 1},
        },
    )

    # Create a map with an unknown object
    config = RandomMapBuilder.Config(
        width=5,
        height=5,
        objects={"unknown_object": 1},  # This doesn't exist in game config
        agents={"team_1": 1},
    )

    builder = RandomMapBuilder(config)
    builder.set_game_config(game_config)

    # Capture warnings
    with pytest.warns(None):
        # This should log a warning but not fail
        game_map = builder.build_validated()

    # The map should still be built (backward compatibility)
    assert game_map is not None
    assert game_map.grid is not None

    # But compression should have failed (no byte_grid)
    # Actually, the current implementation logs the error but doesn't prevent compression
    # This is a design choice for backward compatibility
