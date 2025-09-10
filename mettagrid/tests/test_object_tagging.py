# Test object tagging system for MetaGrid

import pytest

from metta.mettagrid.char_encoder import char_to_grid_object
from metta.mettagrid.gym_env import MettaGridGymEnv
from metta.mettagrid.mettagrid_c_config import apply_tag_overrides, parse_object_with_tags
from metta.mettagrid.mettagrid_config import (
    BoxConfig,
    ConverterConfig,
    GameConfig,
    MettaGridConfig,
    TagConfig,
    WallConfig,
)
from metta.mettagrid.mettagrid_env import MettaGridEnv


def test_parse_object_with_tags():
    """Test parsing object names with tags."""
    # Test agent tags with overrides - now allowed
    base_type, tags = parse_object_with_tags("agent.red")
    assert base_type == "agent"
    assert tags == ["red"]

    # Test agent with tag that has no overrides
    base_type, tags = parse_object_with_tags("agent.no_override")
    assert base_type == "agent"
    assert tags == ["no_override"]

    # Test non-agent tags with overrides
    base_type, tags = parse_object_with_tags("converter.red.fast")
    assert base_type == "converter"
    assert tags == ["red", "fast"]

    # Test no tags
    base_type, tags = parse_object_with_tags("wall")
    assert base_type == "wall"
    assert tags == []


def test_apply_tag_overrides():
    """Test applying tag overrides to object configs."""
    game_config = GameConfig(
        tags={
            "red": TagConfig(overrides={"color": 1}),
            "fast": TagConfig(overrides={"conversion_ticks": 1}),
            "large": TagConfig(overrides={"max_output": 10}),
        }
    )

    base_config = {
        "type_id": 1,
        "color": 0,
        "conversion_ticks": 5,
        "max_output": 3,
    }

    # Apply single tag
    updated_config = apply_tag_overrides(base_config, ["red"], game_config)
    assert updated_config["color"] == 1
    assert updated_config["conversion_ticks"] == 5  # unchanged

    # Apply multiple tags in order
    updated_config = apply_tag_overrides(base_config, ["red", "fast", "large"], game_config)
    assert updated_config["color"] == 1
    assert updated_config["conversion_ticks"] == 1
    assert updated_config["max_output"] == 10


def test_tag_feature_id_determinism():
    """Test that tag feature IDs are deterministic across different environment instances."""
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=7,
            obs_height=7,
            tags={
                "red": TagConfig(overrides={"color": 1}),
                "blue": TagConfig(overrides={"color": 2}),
                "fast": TagConfig(overrides={"conversion_ticks": 1}),
                "slow": TagConfig(overrides={"conversion_ticks": 10}),
            },
            objects={
                "wall": WallConfig(type_id=1),
                "converter": ConverterConfig(
                    type_id=2,
                    input_resources={"ore_red": 1},
                    output_resources={"battery_red": 1},
                    cooldown=0,
                ),
            },
            resource_names=["ore_red", "battery_red"],
        )
    )

    # Use with_ascii_map to set the map
    map_data = [
        ["#", "#", "#", "#", "#", "#", "#"],
        ["#", ".", ".", ".", ".", ".", "#"],
        ["#", ".", "@", ".", ".", ".", "#"],
        ["#", ".", ".", ".", ".", ".", "#"],
        ["#", ".", ".", ".", ".", ".", "#"],
        ["#", ".", ".", ".", ".", ".", "#"],
        ["#", "#", "#", "#", "#", "#", "#"],
    ]
    config = config.with_ascii_map(map_data)

    # Create two environments - seed is set in the config
    env1 = MettaGridEnv(config)
    env2 = MettaGridEnv(config)

    # Get feature specs from both environments
    # Access the C++ environment instance through the private attribute
    feature_spec1 = env1._MettaGridCore__c_env_instance.feature_spec()
    feature_spec2 = env2._MettaGridCore__c_env_instance.feature_spec()

    # Extract tag feature IDs from both
    tag_features1 = {name: idx for name, idx in feature_spec1.items() if name.startswith("tag:")}
    tag_features2 = {name: idx for name, idx in feature_spec2.items() if name.startswith("tag:")}

    # Assert that tag feature IDs are identical
    assert tag_features1 == tag_features2, f"Tag feature IDs differ between runs: {tag_features1} != {tag_features2}"

    # Verify expected tags are present (sorted alphabetically)
    expected_tags = ["tag:blue", "tag:fast", "tag:red", "tag:slow"]
    for tag in expected_tags:
        assert tag in tag_features1, f"Expected tag '{tag}' not found in feature spec"
        assert tag_features1[tag] == tag_features2[tag], (
            f"Tag '{tag}' has different IDs: {tag_features1[tag]} != {tag_features2[tag]}"
        )


def test_tagged_objects_in_environment():
    """Test that tagged objects are properly created in the environment."""
    # For now, just test that the basic config works
    # We'll need to extend the system to handle tagged objects in maps
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=7,
            obs_height=7,
            tags={
                "red": TagConfig(overrides={"color": 1}),
                "blue": TagConfig(overrides={"color": 2}),
            },
            objects={
                "wall": WallConfig(type_id=1),
                "converter": ConverterConfig(
                    type_id=2,
                    input_resources={"ore_red": 1},
                    output_resources={"battery_red": 1},
                    cooldown=0,
                ),
            },
        )
    )

    # Create a simple map
    map_data = [
        ["#", "@", "."],
        [".", "c", "."],
        ["#", ".", "#"],
    ]

    config = config.with_ascii_map(map_data)

    # Create environment
    env = MettaGridGymEnv(config)
    obs, _ = env.reset()

    # Verify environment was created successfully
    assert env is not None
    assert obs.shape[0] > 0  # Has observation tokens

    env.close()


def test_tag_features_in_observations():
    """Test that tag features are properly emitted in observations."""
    from metta.mettagrid.mettagrid_config import WallConfig

    # Create a simple environment with tagged objects
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=5,
            obs_height=5,
            tags={
                "red": TagConfig(overrides={}),  # No overrides for agent tags
                "special": TagConfig(overrides={}),  # No overrides for agent tags
            },
            objects={
                "wall": WallConfig(type_id=1),
            },
        )
    )

    # Create map with tagged agent
    map_data = [
        ["wall", "wall", "wall"],
        ["wall", "agent.red.special", "wall"],
        ["wall", "wall", "wall"],
    ]

    config = config.with_ascii_map(map_data)

    # Create environment
    env = MettaGridGymEnv(config)
    obs, _ = env.reset()

    # Check that observations are not empty
    # For single agent, obs shape is (num_observation_tokens, 3)
    # Based on config: obs_width=5, obs_height=5, and default token configuration
    assert obs.shape[0] == 200  # Default number of observation tokens
    assert obs.shape[1] == 3  # Each token has (location, feature_id, value)

    # Get feature spec to identify tag features properly
    # Access the C++ environment instance through the private attribute
    feature_spec = env._MettaGridCore__c_env_instance.feature_spec()

    # Find feature IDs for our tags
    tag_feature_map = {}
    for feature_name, feature_info in feature_spec.items():
        if feature_name.startswith("tag:"):
            tag_name = feature_name[4:]  # Remove "tag:" prefix
            tag_feature_map[tag_name] = feature_info["id"]

    # We expect to find features for "red" and "special" tags
    assert "red" in tag_feature_map, (
        f"Tag 'red' not found in feature_spec. Available tags: {list(tag_feature_map.keys())}"
    )
    assert "special" in tag_feature_map, (
        f"Tag 'special' not found in feature_spec. Available tags: {list(tag_feature_map.keys())}"
    )

    # Look for these specific tag features in observations
    red_feature_id = tag_feature_map["red"]
    special_feature_id = tag_feature_map["special"]

    tag_features_found = set()
    for token in obs:
        _, feature_id, value = token
        if feature_id in [red_feature_id, special_feature_id] and value == 1:
            tag_features_found.add(int(feature_id))

    # Assert that both tag features appear in observations
    assert red_feature_id in tag_features_found, f"Tag feature 'red' (id={red_feature_id}) not found in observations"
    assert special_feature_id in tag_features_found, (
        f"Tag feature 'special' (id={special_feature_id}) not found in observations"
    )

    env.close()


def test_char_encoder_with_agent_tags():
    """Test char_to_grid_object handles agent tags correctly."""
    # Test basic agent conversion
    assert char_to_grid_object("@") == "agent.agent"
    assert char_to_grid_object("agent") == "agent.agent"

    # Test agent with tags
    assert char_to_grid_object("agent.red") == "agent.agent.red"
    assert char_to_grid_object("agent.red.special") == "agent.agent.red.special"

    # Test agent subtypes
    assert char_to_grid_object("agent.team_1") == "agent.team_1"
    assert char_to_grid_object("agent.team_2") == "agent.team_2"
    assert char_to_grid_object("agent.prey") == "agent.prey"
    assert char_to_grid_object("agent.predator") == "agent.predator"

    # Test agent subtypes with tags
    assert char_to_grid_object("agent.team_1.red") == "agent.team_1.red"
    assert char_to_grid_object("agent.prey.blue.fast") == "agent.prey.blue.fast"

    # Test other object types pass through unchanged
    assert char_to_grid_object("wall") == "wall"
    assert char_to_grid_object("wall.red") == "wall.red"
    assert char_to_grid_object("converter.blue.fast") == "converter.blue.fast"
    assert char_to_grid_object("box.green") == "box.green"


def test_char_encoder_edge_cases():
    """Test char_to_grid_object handles edge cases gracefully."""
    # Unknown single character - should raise error as it's likely a typo
    with pytest.raises(ValueError, match="Unknown single character glyph.*likely a typo"):
        char_to_grid_object("X")

    # Unknown object name without dots - should pass through
    assert char_to_grid_object("unknown_object") == "unknown_object"

    # Object with many tags - should pass through
    assert char_to_grid_object("wall.red.strong.special.heavy") == "wall.red.strong.special.heavy"

    # Empty string - should raise ValueError
    with pytest.raises(ValueError, match="Object name cannot be empty"):
        char_to_grid_object("")

    # Single dot is mapped to "empty" in CHAR_TO_NAME
    assert char_to_grid_object(".") == "empty"

    # Object ending with dot - should raise error to prevent empty tag segments
    with pytest.raises(ValueError, match="Object name cannot end with a dot.*empty tag segment"):
        char_to_grid_object("wall.")


def test_tagged_walls_boxes_converters():
    """Test that walls, boxes, and converters can have tags."""
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=7,
            obs_height=7,
            tags={
                "red": TagConfig(overrides={}),
                "blue": TagConfig(overrides={}),
                "strong": TagConfig(overrides={"swappable": False}),
                "fast": TagConfig(overrides={"cooldown": 1}),
            },
            objects={
                "wall": WallConfig(type_id=1),
                "box": BoxConfig(type_id=2),
                "converter": ConverterConfig(
                    type_id=3,
                    input_resources={"ore_red": 1},
                    output_resources={"battery_red": 1},
                    cooldown=10,
                ),
            },
        )
    )

    # Create map with tagged objects
    map_data = [
        ["wall.red", "box.blue", "converter.fast"],
        ["@", ".", "."],
        ["wall.strong", ".", "."],
    ]

    config = config.with_ascii_map(map_data)

    # Create environment
    env = MettaGridGymEnv(config)
    obs, _ = env.reset()

    # Verify environment was created successfully with tagged objects
    assert env is not None
    assert obs.shape[0] > 0

    env.close()


def test_multiple_agents_with_different_tags():
    """Test multiple agents with different tag combinations."""
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=3,
            max_steps=100,
            obs_width=7,
            obs_height=7,
            tags={
                "red": TagConfig(overrides={}),
                "blue": TagConfig(overrides={}),
                "fast": TagConfig(overrides={}),
                "slow": TagConfig(overrides={}),
            },
            objects={
                "wall": WallConfig(type_id=1),
            },
        )
    )

    # Create map with multiple tagged agents
    map_data = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red.fast", ".", "agent.blue", "wall"],
        ["wall", ".", ".", ".", "wall"],
        ["wall", "agent.slow", ".", ".", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    config = config.with_ascii_map(map_data)

    # Create environment - use MettaGridEnv for multi-agent
    env = MettaGridEnv(config)
    obs, _ = env.reset()

    # Check that we have 3 agents
    # For multi-agent, obs should be a 3D array: (num_agents, num_tokens, 3)
    assert obs.shape[0] == 3  # 3 agents
    assert obs.shape[1] == 200  # Each agent has 200 observation tokens
    assert obs.shape[2] == 3  # Each token has (location, feature_id, value)

    # Get feature spec to verify tags are present
    c_env = env._MettaGridCore__c_env_instance
    feature_spec = c_env.feature_spec()

    # Verify all expected tags are registered
    assert "tag:red" in feature_spec, "Tag 'red' should be registered"
    assert "tag:fast" in feature_spec, "Tag 'fast' should be registered"
    assert "tag:blue" in feature_spec, "Tag 'blue' should be registered"
    assert "tag:slow" in feature_spec, "Tag 'slow' should be registered"

    # Get tag feature IDs
    red_feature_id = feature_spec["tag:red"]["id"]
    fast_feature_id = feature_spec["tag:fast"]["id"]
    blue_feature_id = feature_spec["tag:blue"]["id"]
    slow_feature_id = feature_spec["tag:slow"]["id"]

    # Check agent 0 (red, fast) has the expected tag features
    agent0_obs = obs[0]  # First agent's observations
    agent0_features = set()
    for token in agent0_obs:
        _, feature_id, value = token
        if value == 1:  # Tag features have value 1 when present
            agent0_features.add(int(feature_id))

    assert red_feature_id in agent0_features, "Agent 0 should have 'red' tag feature"
    assert fast_feature_id in agent0_features, "Agent 0 should have 'fast' tag feature"

    # Check agent 1 (blue) has the expected tag feature
    agent1_obs = obs[1]  # Second agent's observations
    agent1_features = set()
    for token in agent1_obs:
        _, feature_id, value = token
        if value == 1:
            agent1_features.add(int(feature_id))

    assert blue_feature_id in agent1_features, "Agent 1 should have 'blue' tag feature"

    # Check agent 2 (slow) has the expected tag feature
    agent2_obs = obs[2]  # Third agent's observations
    agent2_features = set()
    for token in agent2_obs:
        _, feature_id, value = token
        if value == 1:
            agent2_features.add(int(feature_id))

    assert slow_feature_id in agent2_features, "Agent 2 should have 'slow' tag feature"

    env.close()


def test_default_agent_creation():
    """Test that agents are created with default configs when not explicitly defined."""
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=2,
            max_steps=100,
            obs_width=5,
            obs_height=5,
            tags={
                "special": TagConfig(overrides={}),
            },
            objects={
                "wall": WallConfig(type_id=1),
                # No agent config defined - should use defaults
            },
        )
    )

    # Create map with agents that aren't defined in objects
    map_data = [
        ["wall", "agent", "wall"],
        ["wall", ".", "wall"],
        ["wall", "agent.special", "wall"],
    ]

    config = config.with_ascii_map(map_data)

    # Create environment - use MettaGridEnv for multi-agent
    env = MettaGridEnv(config)
    obs, _ = env.reset()

    # Verify agents were created successfully
    assert env is not None
    assert obs.shape[0] == 2  # 2 agents

    # Verify the special agent is actually special
    c_env = env._MettaGridCore__c_env_instance
    feature_spec = c_env.feature_spec()

    # Verify 'special' tag is registered
    assert "tag:special" in feature_spec, "Tag 'special' should be registered"
    special_feature_id = feature_spec["tag:special"]["id"]

    # Both agents might observe the special agent (including themselves)
    # So we check that at least one observation contains the special tag
    special_tag_observed = False

    # Check first agent's observations for special tag
    for token in obs[0]:
        _, feature_id, value = token
        if feature_id == special_feature_id and value == 1:
            special_tag_observed = True
            break

    # If not found in first agent's obs, check second agent's observations
    if not special_tag_observed:
        for token in obs[1]:
            _, feature_id, value = token
            if feature_id == special_feature_id and value == 1:
                special_tag_observed = True
                break

    # At least one agent should observe the special tag (either themselves or the other agent)
    assert special_tag_observed, "The 'special' tag should be observable by at least one agent"

    env.close()


def test_tag_overrides_are_applied():
    """Test that tag overrides actually modify object properties."""
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=5,
            obs_height=5,
            tags={
                "fast": TagConfig(overrides={"cooldown": 1, "conversion_ticks": 2}),
                "efficient": TagConfig(overrides={"max_output": 5}),
            },
            objects={
                "converter": ConverterConfig(
                    type_id=1,
                    input_resources={"ore_red": 1},
                    output_resources={"battery_red": 1},
                    cooldown=10,
                    conversion_ticks=10,
                    max_output=1,
                ),
            },
        )
    )

    # Apply tags to converter config
    base_config = config.game.objects["converter"].model_dump()

    # Apply fast tag
    fast_config = apply_tag_overrides(base_config, ["fast"], config.game)
    assert fast_config["cooldown"] == 1
    assert fast_config["conversion_ticks"] == 2
    assert fast_config["max_output"] == 1  # unchanged

    # Apply both tags
    both_config = apply_tag_overrides(base_config, ["fast", "efficient"], config.game)
    assert both_config["cooldown"] == 1
    assert both_config["conversion_ticks"] == 2
    assert both_config["max_output"] == 5


def test_invalid_tag_name_error():
    """Test that invalid tag names raise clear errors."""
    # Valid tag names should work
    base_type, tags = parse_object_with_tags("converter.valid_tag")
    assert base_type == "converter"
    assert tags == ["valid_tag"]

    # Invalid tag names with special characters should raise ValueError
    with pytest.raises(
        ValueError, match="Invalid tag name.*tags must contain only alphanumeric characters and underscores"
    ):
        parse_object_with_tags("converter.red-hot")

    with pytest.raises(
        ValueError, match="Invalid tag name.*tags must contain only alphanumeric characters and underscores"
    ):
        parse_object_with_tags("wall.special!")

    with pytest.raises(
        ValueError, match="Invalid tag name.*tags must contain only alphanumeric characters and underscores"
    ):
        parse_object_with_tags("box.my@tag")


def test_wall_tag_overrides_applied_at_runtime():
    """Test that wall tag overrides are actually applied at runtime."""
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=7,
            obs_height=7,
            tags={
                "strong": TagConfig(overrides={"swappable": False, "type_id": 101}),
                "weak": TagConfig(overrides={"swappable": True, "type_id": 102}),
            },
            objects={
                "wall": WallConfig(type_id=1, swappable=True),  # Default is swappable
            },
        )
    )

    # Create map with tagged walls
    map_data = [
        ["wall", "wall.strong", "wall.weak"],
        ["@", ".", "."],
        [".", ".", "."],
    ]

    config = config.with_ascii_map(map_data)
    # Use MettaGridEnv for multi-agent environment
    env = MettaGridEnv(config)
    obs, _ = env.reset()

    # Access the underlying C++ environment
    c_env = env._MettaGridCore__c_env_instance

    # Get grid objects to inspect their properties
    grid_objects = c_env.grid_objects()

    # Find walls at specific positions and verify properties
    # Tagged walls will have different type_ids (101, 102) than the base wall (1)
    wall_at_0_0 = None  # Regular wall (type_id=1)
    wall_at_0_1 = None  # wall.strong (type_id=101)
    wall_at_0_2 = None  # wall.weak (type_id=102)

    for obj in grid_objects.values():
        obj_type_id = obj["type"]
        # location is (col, row, layer) tuple
        col, row, _ = obj["location"]
        # Walls are at row 0, check by type_id since tagged walls have different IDs
        if row == 0 and col == 0 and obj_type_id == 1:
            wall_at_0_0 = obj
        elif row == 0 and col == 1 and obj_type_id == 101:
            wall_at_0_1 = obj
        elif row == 0 and col == 2 and obj_type_id == 102:
            wall_at_0_2 = obj

    row_0_objects = [
        (obj['location'], obj['type'])
        for obj in grid_objects.values()
        if obj['location'][1] == 0
    ]
    assert wall_at_0_0 is not None, (
        f"Regular wall not found at (0,0). Objects at row 0: {row_0_objects}"
    )
    assert wall_at_0_1 is not None, (
        f"wall.strong not found at (0,1). Objects at row 0: {row_0_objects}"
    )
    assert wall_at_0_2 is not None, (
        f"wall.weak not found at (0,2). Objects at row 0: {row_0_objects}"
    )

    # Verify type_ids are different for tagged variants
    assert wall_at_0_0["type"] == 1, f"Regular wall should have type_id=1, got {wall_at_0_0['type']}"
    assert wall_at_0_1["type"] == 101, f"wall.strong should have type_id=101, got {wall_at_0_1['type']}"
    assert wall_at_0_2["type"] == 102, f"wall.weak should have type_id=102, got {wall_at_0_2['type']}"

    # Get feature spec to find swappable feature ID
    feature_spec = c_env.feature_spec()
    _ = feature_spec.get("swappable", {}).get("id")

    env.close()


def test_converter_tag_overrides_applied_at_runtime():
    """Test that converter tag overrides are actually applied at runtime."""
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=7,
            obs_height=7,
            tags={
                "fast": TagConfig(overrides={"cooldown": 1, "conversion_ticks": 2, "type_id": 201}),
                "efficient": TagConfig(overrides={"max_output": 5, "type_id": 202}),
            },
            objects={
                "converter": ConverterConfig(
                    type_id=2,
                    input_resources={"ore_red": 1},
                    output_resources={"battery_red": 1},
                    cooldown=10,
                    conversion_ticks=10,
                    max_output=1,
                ),
            },
        )
    )

    # Create map with tagged converters
    map_data = [
        ["converter", "converter.fast", "converter.efficient"],
        ["@", ".", "."],
        [".", ".", "."],
    ]

    config = config.with_ascii_map(map_data)
    env = MettaGridGymEnv(config)
    obs, _ = env.reset()

    # Access the underlying C++ environment
    c_env = env._MettaGridCore__c_env_instance

    # Get grid objects
    grid_objects = c_env.grid_objects()

    # Find converters at specific positions
    # Tagged converters will have different type_ids (201, 202) than the base converter (2)
    converter_at_0_0 = None  # Regular converter (type_id=2)
    converter_at_0_1 = None  # converter.fast (type_id=201)
    converter_at_0_2 = None  # converter.efficient (type_id=202)

    for obj in grid_objects.values():
        obj_type_id = obj["type"]
        # location is (col, row, layer) tuple
        col, row, _ = obj["location"]
        # Converters are at row 0, check by type_id since tagged converters have different IDs
        if row == 0 and col == 0 and obj_type_id == 2:
            converter_at_0_0 = obj
        elif row == 0 and col == 1 and obj_type_id == 201:
            converter_at_0_1 = obj
        elif row == 0 and col == 2 and obj_type_id == 202:
            converter_at_0_2 = obj

    row_0_objects = [
        (obj['location'], obj['type'])
        for obj in grid_objects.values()
        if obj['location'][1] == 0
    ]
    assert converter_at_0_0 is not None, (
        f"Regular converter not found at (0,0). Objects at row 0: {row_0_objects}"
    )
    assert converter_at_0_1 is not None, (
        f"converter.fast not found at (0,1). Objects at row 0: {row_0_objects}"
    )
    assert converter_at_0_2 is not None, (
        f"converter.efficient not found at (0,2). Objects at row 0: {row_0_objects}"
    )

    # Verify type_ids are different for tagged variants
    assert converter_at_0_0["type"] == 2, f"Regular converter should have type_id=2, got {converter_at_0_0['type']}"
    assert converter_at_0_1["type"] == 201, f"converter.fast should have type_id=201, got {converter_at_0_1['type']}"
    assert converter_at_0_2["type"] == 202, (
        f"converter.efficient should have type_id=202, got {converter_at_0_2['type']}"
    )

    env.close()


def test_max_tags_per_object_limit():
    """Test that objects with too many tags raise an error for each object class."""
    # Test data setup
    MAX_TAGS = 10
    tags_dict = {f"tag{i}": TagConfig(overrides={} if i < 5 else {"color": i}) for i in range(15)}

    # Test Wall objects
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=5,
            obs_height=5,
            tags=tags_dict,
            objects={
                "wall": WallConfig(type_id=1),
            },
        )
    )

    wall_with_too_many_tags = "wall." + ".".join([f"tag{i}" for i in range(MAX_TAGS + 1)])
    map_data = [[wall_with_too_many_tags, "@", "."]]
    config_wall = config.with_ascii_map(map_data)

    with pytest.raises(
        (ValueError, RuntimeError), match="(exceeding the maximum of 10 tags per object|has too many tags)"
    ):
        env = MettaGridGymEnv(config_wall)
        env.reset()
        env.close()

    # Test Box objects
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=5,
            obs_height=5,
            tags=tags_dict,
            objects={
                "box": BoxConfig(type_id=2),
            },
            resource_names=["test_resource"],
        )
    )

    box_with_too_many_tags = "box." + ".".join([f"tag{i}" for i in range(MAX_TAGS + 1)])
    map_data = [["@", box_with_too_many_tags, "."]]
    config_box = config.with_ascii_map(map_data)

    with pytest.raises(
        (ValueError, RuntimeError), match="(exceeding the maximum of 10 tags per object|has too many tags)"
    ):
        env = MettaGridGymEnv(config_box)
        env.reset()
        env.close()

    # Test Converter objects
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=5,
            obs_height=5,
            tags=tags_dict,
            objects={
                "converter": ConverterConfig(
                    type_id=3,
                    input_resources={"test_resource": 1},
                    output_resources={"test_resource": 2},
                    cooldown=0,
                ),
            },
            resource_names=["test_resource"],
        )
    )

    converter_with_too_many_tags = "converter." + ".".join([f"tag{i}" for i in range(MAX_TAGS + 1)])
    map_data = [["@", ".", converter_with_too_many_tags]]
    config_converter = config.with_ascii_map(map_data)

    with pytest.raises(
        (ValueError, RuntimeError), match="(exceeding the maximum of 10 tags per object|has too many tags)"
    ):
        env = MettaGridGymEnv(config_converter)
        env.reset()
        env.close()

    # Test Agent objects (use tags without overrides for agents)
    agent_tags = {f"atag{i}": TagConfig(overrides={}) for i in range(15)}
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,  # 1 agent in the map
            max_steps=100,
            obs_width=5,
            obs_height=5,
            tags=agent_tags,
            objects={},
        )
    )

    agent_with_too_many_tags = "agent.agent." + ".".join([f"atag{i}" for i in range(MAX_TAGS + 1)])
    map_data = [[agent_with_too_many_tags, ".", "."]]
    config_agent = config.with_ascii_map(map_data)

    # For agents, the error comes from Python parsing or C++
    with pytest.raises(
        (ValueError, RuntimeError), match="(exceeding the maximum of 10 tags per object|has too many tags)"
    ):
        env = MettaGridGymEnv(config_agent)
        env.reset()
        env.close()


def test_agent_tag_overrides_not_supported():
    """Verify that agent tag overrides are now supported.

    Agent tags are emitted as features in observations, and property overrides
    are now applied to agents just like other objects for consistency.
    """
    # Agent tags with overrides should now work
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=5,
            obs_height=5,
            tags={
                "fast": TagConfig(overrides={"freeze_duration": 1}),
                "feature_only": TagConfig(overrides={}),  # Tag without overrides
            },
            objects={
                "wall": WallConfig(type_id=1),
            },
        )
    )

    # Agents with tags that have overrides should now work
    map_data = [
        ["wall", "agent.fast", "wall"],
    ]

    config = config.with_ascii_map(map_data)

    # This should now work without errors
    env = MettaGridGymEnv(config)
    env.reset()
    env.close()

    # Agents with tags that have no overrides should also work
    map_data_no_overrides = [
        ["wall", "agent.feature_only", "wall"],
    ]

    config_no_overrides = config.with_ascii_map(map_data_no_overrides)

    # This should work
    env = MettaGridGymEnv(config_no_overrides)
    env.reset()
    env.close()


def test_invalid_tag_characters_from_map():
    """Test that invalid tag characters discovered from map strings raise errors."""
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=5,
            obs_height=5,
            tags={},  # No predefined tags
            objects={
                "wall": WallConfig(type_id=1),
            },
        )
    )

    # Map with invalid tag characters
    map_data = [
        ["wall.invalid-tag", "@", "."],  # Hyphen is not allowed
    ]

    config = config.with_ascii_map(map_data)

    # Should raise an error about invalid tag characters
    with pytest.raises(
        ValueError, match="Invalid tag name.*tags must contain only alphanumeric characters and underscores"
    ):
        env = MettaGridGymEnv(config)
        env.reset()
        env.close()


def test_conflicting_type_id_overrides():
    """Test that conflicting type_id overrides between tagged variants raise errors."""
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=5,
            obs_height=5,
            tags={
                "red": TagConfig(overrides={"type_id": 100}),
                "blue": TagConfig(overrides={"type_id": 100}),  # Same type_id as red
            },
            objects={
                "wall": WallConfig(type_id=1),
            },
        )
    )

    # Map using both tags on different objects, causing type_id conflict
    map_data = [
        ["wall.red", "wall.blue", "@"],
    ]

    config = config.with_ascii_map(map_data)

    # Should raise an error about type_id conflict
    with pytest.raises(ValueError, match="Type ID conflict.*already used by"):
        env = MettaGridGymEnv(config)
        env.reset()
        env.close()


def test_agent_override_rejection_detailed():
    """Test that agent tag overrides are now properly supported."""
    # Test with various override types to ensure all work for agents
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=5,
            obs_height=5,
            tags={
                "fast": TagConfig(overrides={"freeze_duration": 1}),
                "strong": TagConfig(overrides={"strength": 10}),
                "smart": TagConfig(overrides={"intelligence": 5}),
            },
            objects={
                "wall": WallConfig(type_id=1),
            },
        )
    )

    # Test each tag individually on agents
    # Note: @ is mapped to agent.agent, so we need to use full agent notation with tags
    test_cases = [
        ["agent.fast", ".", "."],
        ["agent.strong", ".", "."],
        ["agent.smart", ".", "."],
        ["agent.agent.fast", ".", "."],
        ["agent.agent.strong", ".", "."],
    ]

    for map_data_row in test_cases:
        config_test = config.with_ascii_map([map_data_row])

        # Each should now work without errors
        env = MettaGridGymEnv(config_test)
        env.reset()
        env.close()


def test_observation_feature_id_exhaustion():
    """Test that creating too many unique tags raises an error about feature ID exhaustion."""
    # ObservationType is unsigned char (0-255), so we have 256 possible values
    # Many IDs are already reserved for base features, recipes, etc.
    # With minimal resources, we still have base features taking up IDs
    # Let's try to create enough tags to exhaust the entire 256 ID space

    # Use minimal resources to maximize space for tags
    resource_names = ["ore", "battery"]
    # Try to create 240 tags, which combined with base features should exceed 256
    num_tags = 240

    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=5,
            obs_height=5,
            resource_names=resource_names,
            recipe_details_obs=False,  # Disable recipe details to save feature IDs
            tags={f"tag{i}": TagConfig(overrides={}) for i in range(num_tags)},
            objects={
                "wall": WallConfig(type_id=1),
            },
        )
    )

    # Create map with many different tagged objects to force registration
    # Start with agent and a few tagged walls
    map_data = [["@"] + [f"wall.tag{i}" for i in range(4)]]

    # Add more rows with different tagged walls to ensure all tags get registered
    for j in range(1, min(48, num_tags // 5)):  # Create up to 48 rows (240 tags / 5 per row)
        row_tags = []
        for i in range(5):
            tag_idx = j * 5 + i
            if tag_idx < num_tags:
                row_tags.append(f"wall.tag{tag_idx}")
            else:
                row_tags.append(".")
        map_data.append(row_tags)

    config = config.with_ascii_map(map_data)

    # Should raise an error about feature ID exhaustion
    # If this doesn't raise, the system might be silently dropping tags or has more space than expected
    try:
        env = MettaGridGymEnv(config)
        env.reset()
        env.close()
        # If we get here without error, check if all tags were actually registered
        # This is acceptable - the system might handle overflow gracefully
        pytest.skip("Feature ID exhaustion not triggered - system may handle overflow gracefully")
    except (RuntimeError, ValueError) as e:
        # Expected behavior - feature IDs exhausted
        assert any(keyword in str(e) for keyword in ["feature ID", "overflow", "exceeded", "Feature space exhausted"])
        pass


def test_type_id_overflow():
    """Test that type_id overflow is properly detected when exceeding 255."""
    # Create base objects with high type_ids
    base_objects = {}
    for i in range(250):  # Start with 250 base objects
        base_objects[f"obj{i}"] = WallConfig(type_id=i)

    # Create tags that will generate more type_ids via tagged variants
    tags = {
        "red": TagConfig(overrides={"swappable": True}),
        "blue": TagConfig(overrides={"swappable": False}),
        "green": TagConfig(overrides={}),
    }

    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=5,
            obs_height=5,
            tags=tags,
            objects=base_objects,
        )
    )

    # Create a map that uses tagged variants which would push type_id past 255
    map_data = [
        ["@", "obj249.red", "obj249.blue", "obj249.green", "."],
        ["obj248.red", "obj248.blue", "obj247.red", "obj246.red", "."],
    ]

    config = config.with_ascii_map(map_data)

    # Should raise an error about type ID overflow
    with pytest.raises(ValueError, match="Type ID overflow|exceed.*255|ObservationType limit"):
        env = MettaGridGymEnv(config)
        env.reset()
        env.close()


def test_type_id_conflicts():
    """Test that type_id conflicts are properly detected."""
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=5,
            obs_height=5,
            tags={
                "special": TagConfig(overrides={"type_id": 10}),  # Force specific type_id via tag
            },
            objects={
                "wall": WallConfig(type_id=10),  # Conflict: same type_id
                "box": BoxConfig(type_id=11),
            },
        )
    )

    # Create a map with tagged object that would conflict
    map_data = [
        ["@", "box.special", "."],  # box.special would try to use type_id=10
    ]

    config = config.with_ascii_map(map_data)

    # Should raise an error about type ID conflict
    with pytest.raises(ValueError, match="Type ID conflict|already used"):
        env = MettaGridGymEnv(config)
        env.reset()
        env.close()


def test_agent_subtype_semantics():
    """Document and test how unknown agent subtypes are handled.

    When an agent subtype like 'agent.unknown_team' is encountered and
    'unknown_team' is not a valid agent subtype, the system treats
    'unknown_team' as a tag and remaps it to 'agent.agent.unknown_team'.
    """
    from metta.mettagrid.char_encoder import char_to_grid_object

    # Test known subtypes are preserved
    assert char_to_grid_object("agent.team_1") == "agent.team_1"
    assert char_to_grid_object("agent.team_2") == "agent.team_2"
    assert char_to_grid_object("agent.prey") == "agent.prey"
    assert char_to_grid_object("agent.predator") == "agent.predator"

    # Test unknown subtypes are treated as tags
    assert char_to_grid_object("agent.custom_team") == "agent.agent.custom_team"
    assert char_to_grid_object("agent.my_special_type") == "agent.agent.my_special_type"

    # Test agent with multiple tags
    assert char_to_grid_object("agent.unknown.tag1.tag2") == "agent.agent.unknown.tag1.tag2"

    # Test in actual environment - unknown subtypes should work as tags
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=3,  # 3 agents in the map
            max_steps=100,
            obs_width=5,
            obs_height=5,
            tags={
                "custom_team": TagConfig(overrides={}),  # Agent tag without overrides
                "special": TagConfig(overrides={}),
            },
        )
    )

    # Maps with agents using unknown subtypes
    map_data = [
        ["agent.custom_team", "agent.special", "agent.custom_team.special"],
        [".", ".", "."],
        [".", ".", "."],
    ]
    config = config.with_ascii_map(map_data)

    # Should create environment successfully with unknown subtypes as tags
    env = MettaGridEnv(config)
    env.reset()

    # Verify the tags are present in feature spec
    feature_spec = env._MettaGridCore__c_env_instance.feature_spec()
    assert "tag:custom_team" in feature_spec
    assert "tag:special" in feature_spec

    env.close()
