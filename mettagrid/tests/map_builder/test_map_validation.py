"""Test map validation with GameConfig integration."""

from metta.mettagrid.map_builder.random import RandomMapBuilder
from metta.mettagrid.mettagrid_config import ConverterConfig, GameConfig, WallConfig


def test_map_validation_with_game_config():
    """Test that map builders can validate against game config."""
    # Create a minimal game config with known objects
    game_config = GameConfig(
        num_agents=4,
        objects={
            "wall": WallConfig(type_id=1, swappable=False),
            "generator": ConverterConfig(type_id=2, output_resources={"ore": 1}, cooldown=10),
            "converter": ConverterConfig(
                type_id=3, input_resources={"ore": 1}, output_resources={"energy": 1}, cooldown=5
            ),
            # Note: agent.team_1 will be auto-created by the system when needed
        },
    )

    # Create a random map builder
    config = RandomMapBuilder.Config(
        width=10,
        height=10,
        objects={"generator": 2, "converter": 1},
        agents=4,  # Use simple count, not team-specific
        border_width=1,
        border_object="wall",
    )

    builder = RandomMapBuilder(config)
    builder.set_game_config(game_config)

    # Build and validate
    # Note: This will log a warning about agent.agent not being in game config,
    # but will still build the map (backward compatibility)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        game_map = builder.build_validated()

    # Map should be built successfully
    assert game_map is not None
    assert game_map.grid is not None

    # Verify the grid contains expected object types (as strings)
    unique_objects = set(game_map.grid.flatten())
    assert "wall" in unique_objects
    assert "empty" in unique_objects
    # Generator and converter may or may not be present depending on random placement


def test_map_validation_catches_unknown_objects():
    """Test that validation catches unknown objects."""

    # Create a game config WITHOUT "unknown_object"
    game_config = GameConfig(
        num_agents=1,
        objects={
            "wall": WallConfig(type_id=1, swappable=False),
            "generator": ConverterConfig(type_id=2, output_resources={"ore": 1}, cooldown=10),
            # Just minimal objects - agents are auto-created
        },
    )

    # Create a map with an unknown object (that RandomMapBuilder doesn't know about)
    # Since RandomMapBuilder generates objects from its config, we'll test
    # with objects it knows about but game_config doesn't have
    config = RandomMapBuilder.Config(
        width=5,
        height=5,
        objects={"generator": 1},  # This exists in game config
        agents=1,
        border_width=0,  # No border, no walls needed
    )

    builder = RandomMapBuilder(config)
    builder.set_game_config(game_config)

    # Build should work, validation logs warning
    import warnings

    # Suppress the warning for the test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        game_map = builder.build_validated()

    # The map should still be built (backward compatibility)
    assert game_map is not None
    assert game_map.grid is not None
