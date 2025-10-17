# Test tag system functionality for mettagrid
import pytest

from mettagrid.config.mettagrid_c_config import convert_to_cpp_game_config
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    ConverterConfig,
    GameConfig,
    MettaGridConfig,
    WallConfig,
)
from mettagrid.core import MettaGridCore
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.mapgen.utils.ascii_grid import DEFAULT_CHAR_TO_NAME
from mettagrid.test_support import TokenTypes

NUM_OBS_TOKENS = 200


@pytest.fixture
def env_with_tags() -> MettaGridCore:
    """Create an environment with objects that have tags."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=2,
            max_steps=1000,
            obs_width=5,
            obs_height=5,
            num_observation_tokens=NUM_OBS_TOKENS,
            actions=ActionsConfig(
                noop=ActionConfig(),
                move=ActionConfig(),
            ),
            objects={
                "wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID, tags=["solid", "blocking"]),
            },
            resource_names=[],
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#", "#", "#", "#", "#"],
                    ["#", "@", ".", ".", ".", ".", "#"],
                    ["#", ".", ".", ".", ".", ".", "#"],
                    ["#", ".", ".", ".", ".", ".", "#"],
                    ["#", "@", ".", ".", ".", ".", "#"],
                    ["#", "#", "#", "#", "#", "#", "#"],
                ],
                char_to_name_map=DEFAULT_CHAR_TO_NAME,
            ),
        )
    )

    return MettaGridCore(cfg)


@pytest.fixture
def env_with_duplicate_tags() -> MettaGridCore:
    """Create an environment where multiple objects share tags."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=1000,
            obs_width=3,
            obs_height=3,
            num_observation_tokens=NUM_OBS_TOKENS,
            actions=ActionsConfig(
                noop=ActionConfig(),
                move=ActionConfig(),
            ),
            agents=[
                AgentConfig(tags=["mobile", "shared_tag"]),
            ],
            objects={
                "wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID, tags=["solid", "shared_tag"]),
            },
            resource_names=[],
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#", "#", "#"],
                    ["#", "@", ".", ".", "#"],
                    ["#", ".", ".", ".", "#"],
                    ["#", ".", ".", ".", "#"],
                    ["#", "#", "#", "#", "#"],
                ],
                char_to_name_map=DEFAULT_CHAR_TO_NAME,
            ),
        )
    )

    return MettaGridCore(cfg)


class TestTags:
    """Test tag system functionality."""

    def test_tags_in_config(self, env_with_tags):
        """Test that tags are properly configured in the game config."""
        obs, _ = env_with_tags.reset()

        # Verify environment creates successfully with tags
        assert obs is not None
        assert len(obs) == 2  # Two agents

        # Check that observation has the expected shape
        assert obs.shape[0] == 2  # Two agents
        assert obs.shape[2] == 3  # Each token is [location, feature, value]

        # Look for walls and their tags in agent 0's observation
        agent0_obs = obs[0]

        # Find walls (type_id = 1) in observation
        wall_locations = set()
        for token in agent0_obs:
            if token[1] == 0 and token[2] == TokenTypes.WALL_TYPE_ID:  # TypeId feature with wall value
                wall_locations.add(token[0])

        # Should find walls in the observation
        assert len(wall_locations) > 0, "Should find walls in observation"

        # Get tag feature ID from environment
        tag_feature_id = env_with_tags.c_env.feature_spec()["tag"]["id"]

        # Check for tag features at wall locations
        tag_features = []
        for token in agent0_obs:
            if token[0] in wall_locations and token[1] == tag_feature_id:
                tag_features.append(token[2])  # token[2] contains the tag ID

        # Walls should have tag features
        assert len(tag_features) > 0, "Walls should have tag features"

    def test_tags_in_observations(self, env_with_tags):
        """Test that tags appear in observations with correct IDs."""
        obs, _ = env_with_tags.reset()
        agent0_obs = obs[0]

        # Wall has tags ["solid", "blocking"] which should be sorted alphabetically
        # and assigned IDs starting from 0
        # Sorted: ["blocking", "solid"]
        expected_tag_ids = [0, 1]  # 0, 1

        # Find wall locations first
        wall_locations = set()
        for token in agent0_obs:
            if token[1] == 0 and token[2] == TokenTypes.WALL_TYPE_ID:  # TypeId feature with wall value
                wall_locations.add(token[0])

        # Get tag feature ID from environment
        tag_feature_id = env_with_tags.c_env.feature_spec()["tag"]["id"]

        # Find tag features at wall locations
        found_tags = set()
        for token in agent0_obs:
            if token[0] in wall_locations and token[1] == tag_feature_id:
                found_tags.add(token[2])  # token[2] contains the tag ID

        # Should find both tag IDs
        assert len(found_tags) >= 2, f"Should find at least 2 tag IDs, found {found_tags}"
        # Both expected tag IDs should be present
        for tag_id in expected_tag_ids:
            assert tag_id in found_tags, f"Tag ID {tag_id} should be in observations"

    def test_empty_tags(self):
        """Test that objects with no tags work correctly."""
        # Create environment with objects that have no tags
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                obs_width=3,
                obs_height=3,
                num_observation_tokens=NUM_OBS_TOKENS,
                actions=ActionsConfig(noop=ActionConfig()),
                objects={
                    "wall": WallConfig(
                        type_id=TokenTypes.WALL_TYPE_ID,
                        tags=[],  # No tags
                    ),
                },
                resource_names=[],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#"],
                        ["#", "@", "#"],
                        ["#", "#", "#"],
                    ],
                    char_to_name_map=DEFAULT_CHAR_TO_NAME,
                ),
            )
        )

        env = MettaGridCore(cfg)
        obs, _ = env.reset()

        # Environment should work fine with objects that have no tags
        assert obs is not None

        agent_obs = obs[0]

        # Get tag feature ID from environment
        tag_feature_id = env.c_env.feature_spec()["tag"]["id"]

        # Find wall locations
        wall_locations = set()
        for token in agent_obs:
            if token[1] == 0 and token[2] == TokenTypes.WALL_TYPE_ID:  # TypeId feature with wall value
                wall_locations.add(token[0])

        # Check that walls don't have tag tokens
        for token in agent_obs:
            if token[0] in wall_locations and token[1] == tag_feature_id:
                raise AssertionError(f"Wall without tags should not have tag tokens, found tag ID {token[2]}")

    def test_duplicate_tags_across_objects(self, env_with_duplicate_tags):
        """Test that multiple objects can share the same tags."""
        obs, _ = env_with_duplicate_tags.reset()
        assert obs is not None

        agent_obs = obs[0]

        # Both agent and wall have "shared_tag"
        # Agent has ["mobile", "shared_tag"]
        # Wall has ["solid", "shared_tag"]
        # Sorted all unique tags: ["mobile", "shared_tag", "solid"]
        # Expected IDs: mobile=0, shared_tag=1, solid=2
        shared_tag_id = 1  # "shared_tag" should be ID 1

        # Get tag feature ID from environment
        tag_feature_id = env_with_duplicate_tags.c_env.feature_spec()["tag"]["id"]

        # Find wall and agent locations
        wall_locations = set()
        agent_locations = set()

        for token in agent_obs:
            if token[1] == 0:  # TypeId feature
                if token[2] == TokenTypes.WALL_TYPE_ID:
                    wall_locations.add(token[0])
                elif token[2] == 0:  # Agent type ID
                    agent_locations.add(token[0])

        # Find the shared tag in both agent and wall observations
        found_shared_in_wall = False
        found_shared_in_agent = False

        for token in agent_obs:
            if token[1] == tag_feature_id and token[2] == shared_tag_id:
                if token[0] in wall_locations:
                    found_shared_in_wall = True
                if token[0] in agent_locations:
                    found_shared_in_agent = True

        # The shared tag should appear in at least one type of object
        assert found_shared_in_wall or found_shared_in_agent, "Shared tag should be found in observations"

    def test_many_tags_on_single_object(self):
        """Test that an object can have many tags."""
        tags = [f"tag{i:02d}" for i in range(1, 11)]  # tag01 through tag10

        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                obs_width=3,
                obs_height=3,
                num_observation_tokens=NUM_OBS_TOKENS,
                actions=ActionsConfig(noop=ActionConfig()),
                objects={
                    "wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID, tags=tags),
                },
                resource_names=[],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#"],
                        ["#", "@", "#"],
                        ["#", "#", "#"],
                    ],
                    char_to_name_map=DEFAULT_CHAR_TO_NAME,
                ),
            )
        )

        env = MettaGridCore(cfg)
        obs, _ = env.reset()

        # Should handle many tags without issues
        assert obs is not None

        agent_obs = obs[0]

        # Get tag feature ID from environment
        tag_feature_id = env.c_env.feature_spec()["tag"]["id"]

        # Find wall locations
        wall_locations = set()
        for token in agent_obs:
            if token[1] == 0 and token[2] == TokenTypes.WALL_TYPE_ID:  # TypeId feature with wall value
                wall_locations.add(token[0])

        # Count unique tag IDs found on walls
        tag_ids_found = set()
        for token in agent_obs:
            if token[0] in wall_locations and token[1] == tag_feature_id:
                tag_ids_found.add(token[2])  # token[2] contains the tag ID

        # Should find all 10 tags (IDs 0-9 for sorted tags)
        assert len(tag_ids_found) == 10, f"Should find all 10 tags, found {len(tag_ids_found)}"
        # Tag IDs should be consecutive starting from 0
        expected_ids = set(range(0, 10))
        assert tag_ids_found == expected_ids, f"Tag IDs should be {expected_ids}, got {tag_ids_found}"

    def test_tag_id_mapping(self):
        """Test that tag names are consistently mapped to IDs."""
        # Create two environments with same tags in different order
        cfg1 = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                obs_width=3,
                obs_height=3,
                num_observation_tokens=200,
                actions=ActionsConfig(noop=ActionConfig()),
                objects={
                    "wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID, tags=["alpha", "beta"]),
                },
                resource_names=[],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        [".", ".", "."],
                        [".", "@", "."],
                        [".", ".", "#"],  # Wall in bottom-right
                    ],
                    char_to_name_map=DEFAULT_CHAR_TO_NAME,
                ),
            )
        )

        cfg2 = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                obs_width=3,
                obs_height=3,
                num_observation_tokens=200,
                actions=ActionsConfig(noop=ActionConfig()),
                objects={
                    "wall": WallConfig(
                        type_id=TokenTypes.WALL_TYPE_ID,
                        tags=["beta", "alpha"],  # Same tags, different order
                    ),
                },
                resource_names=[],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        [".", ".", "."],
                        [".", "@", "."],
                        [".", ".", "#"],  # Wall in bottom-right
                    ],
                    char_to_name_map=DEFAULT_CHAR_TO_NAME,
                ),
            )
        )

        # Both configs should work and map tags consistently
        env1 = MettaGridCore(cfg1)
        env2 = MettaGridCore(cfg2)

        obs1, _ = env1.reset()
        obs2, _ = env2.reset()

        assert obs1 is not None
        assert obs2 is not None

        # Get tag feature ID from environment
        tag_feature_id = env1.c_env.feature_spec()["tag"]["id"]

        # Extract tag IDs from both environments
        def get_wall_tag_ids(obs):
            # Find wall locations
            wall_locations = set()
            for token in obs:
                if token[1] == 0 and token[2] == TokenTypes.WALL_TYPE_ID:
                    wall_locations.add(token[0])

            # Find tag IDs at wall locations
            tag_ids = set()
            for token in obs:
                if token[0] in wall_locations and token[1] == tag_feature_id:
                    tag_ids.add(token[2])  # token[2] contains the tag ID
            return tag_ids

        tags1 = get_wall_tag_ids(obs1[0])
        tags2 = get_wall_tag_ids(obs2[0])

        # Both should have the same tag IDs (sorted mapping)
        assert tags1 == tags2, f"Tag IDs should be consistent: {tags1} vs {tags2}"
        # Should have exactly 2 tag IDs (alpha and beta)
        assert len(tags1) == 2, f"Should have 2 tag IDs, got {len(tags1)}"
        # Tag IDs should be consecutive starting from 0
        # "alpha" < "beta" alphabetically, so alpha=0, beta=1
        assert tags1 == {0, 1}, f"Expected tag IDs {{0, 1}}, got {tags1}"

    def test_converter_with_tags(self):
        """Test that converter objects can have tags."""
        # Since converters can't be placed via ASCII maps easily,
        # we'll test that the ConverterConfig accepts tags without errors
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                obs_width=3,
                obs_height=3,
                num_observation_tokens=200,
                actions=ActionsConfig(noop=ActionConfig()),
                objects={
                    "converter": ConverterConfig(
                        type_id=2,
                        input_resources={"wood": 1},
                        output_resources={"coal": 1},
                        max_output=10,
                        max_conversions=5,
                        conversion_ticks=10,
                        cooldown=[5],
                        tags=["machine", "converter", "industrial"],
                    ),
                    "wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID, tags=["solid"]),
                },
                resource_names=["wood", "coal"],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#"],
                        ["#", "@", "#"],
                        ["#", "#", "#"],
                    ],
                    char_to_name_map=DEFAULT_CHAR_TO_NAME,
                ),
            )
        )

        # The test verifies that converter config accepts tags without errors
        env = MettaGridCore(cfg)
        obs, _ = env.reset()

        # Get tag feature ID from environment
        tag_feature_id = env.c_env.feature_spec()["tag"]["id"]

        assert obs is not None

        # We can verify walls have their tags to ensure the system works
        agent_obs = obs[0]

        # Find wall locations
        wall_locations = set()
        for token in agent_obs:
            if token[1] == 0 and token[2] == TokenTypes.WALL_TYPE_ID:
                wall_locations.add(token[0])

        # Find tag IDs at wall locations
        wall_tag_ids = set()
        for token in agent_obs:
            if token[0] in wall_locations and token[1] == tag_feature_id:
                wall_tag_ids.add(token[2])  # token[2] contains the tag ID

        # Walls should have the "solid" tag (plus converter and machine tags in the system)
        # All unique tags: ["converter", "industrial", "machine", "solid", "wood", "coal"] (resources might add tags)
        # We should find at least the solid tag
        assert len(wall_tag_ids) >= 1, f"Walls should have at least 1 tag, found {wall_tag_ids}"

    def test_agent_with_tags(self):
        """Test that agents can have tags."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=2,
                max_steps=100,
                obs_width=3,
                obs_height=3,
                num_observation_tokens=200,
                actions=ActionsConfig(noop=ActionConfig()),
                agents=[
                    AgentConfig(team_id=0, tags=["player", "team_red"]),
                    AgentConfig(team_id=1, tags=["player", "team_blue"]),
                ],
                resource_names=[],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        [".", "@", "."],
                        [".", ".", "."],
                        [".", "@", "."],
                    ],
                    char_to_name_map=DEFAULT_CHAR_TO_NAME,
                ),
            )
        )

        env = MettaGridCore(cfg)
        obs, _ = env.reset()

        # Get tag feature ID from environment
        tag_feature_id = env.c_env.feature_spec()["tag"]["id"]

        assert obs is not None
        assert len(obs) == 2  # Two agents

        # Check each agent's observation for agent tags
        for agent_idx in range(2):
            agent_obs = obs[agent_idx]

            # Find agent locations
            agent_locations = set()
            for token in agent_obs:
                if token[1] == 0 and token[2] == 0:  # TypeId feature with agent value
                    agent_locations.add(token[0])

            # Find tag IDs at agent locations
            agent_tag_ids = set()
            for token in agent_obs:
                if token[0] in agent_locations and token[1] == tag_feature_id:
                    agent_tag_ids.add(token[2])  # token[2] contains the tag ID

            # Each agent should have at least 2 tags
            assert len(agent_tag_ids) >= 2, f"Agent {agent_idx} should have at least 2 tags, found {len(agent_tag_ids)}"


def test_tag_id_bounds():
    """Test that tag IDs start at 0 and validate bounds."""
    # Create a config with a few tags
    game_config = GameConfig()
    game_config.agents = [
        AgentConfig(team_id=0, tags=["alpha", "beta", "gamma"]),
        AgentConfig(team_id=1, tags=["delta", "epsilon"]),
    ]

    # Convert and verify tag IDs start at 0
    cpp_config = convert_to_cpp_game_config(game_config)

    # Tag IDs should be: alpha=0, beta=1, delta=2, epsilon=3, gamma=4
    tag_id_map = cpp_config.tag_id_map
    assert len(tag_id_map) == 5

    # Check that tag IDs start at 0
    min_tag_id = min(tag_id_map.keys())
    assert min_tag_id == 0, f"Tag IDs should start at 0, but minimum is {min_tag_id}"

    # Check that tag IDs are sequential
    sorted_ids = sorted(tag_id_map.keys())
    expected_ids = list(range(len(tag_id_map)))
    assert sorted_ids == expected_ids, f"Tag IDs should be sequential from 0, got {sorted_ids}"


def test_too_many_tags_error():
    """Test that having more than 256 tags raises an error."""
    # Create config with 257 unique tags (exceeds uint8 max)
    game_config = GameConfig()

    # Create agents with many unique tags
    tags_per_agent = 50
    num_agents = 6  # 6 * 50 = 300 tags > 256

    game_config.agents = []
    for i in range(num_agents):
        tags = [f"tag_{i}_{j}" for j in range(tags_per_agent)]
        game_config.agents.append(AgentConfig(team_id=i, tags=tags))

    # Should raise ValueError about too many tags
    with pytest.raises(ValueError) as excinfo:
        convert_to_cpp_game_config(game_config)

    assert "Too many unique tags" in str(excinfo.value)
    assert "256" in str(excinfo.value)


def test_team_tag_consistency_enforced():
    """Test that all agents in a team must have identical tags."""
    game_config = GameConfig()

    # Create agents in same team with different tags
    game_config.agents = [
        AgentConfig(team_id=0, tags=["alpha", "beta"]),
        AgentConfig(team_id=0, tags=["alpha", "gamma"]),  # Different tags, same team
        AgentConfig(team_id=1, tags=["delta"]),
    ]

    # Should raise ValueError about inconsistent tags in team
    with pytest.raises(ValueError) as excinfo:
        convert_to_cpp_game_config(game_config)

    assert "All agents in team" in str(excinfo.value)
    assert "must have identical tags" in str(excinfo.value)


def test_team_tag_consistency_success():
    """Test that agents in same team with identical tags work correctly."""
    game_config = GameConfig()

    # Create agents in same team with identical tags
    game_config.agents = [
        AgentConfig(team_id=0, tags=["alpha", "beta"]),
        AgentConfig(team_id=0, tags=["alpha", "beta"]),  # Same tags, same team - OK
        AgentConfig(team_id=1, tags=["gamma", "delta"]),
        AgentConfig(team_id=1, tags=["gamma", "delta"]),  # Same tags, same team - OK
    ]

    # Should succeed
    cpp_config = convert_to_cpp_game_config(game_config)

    # Verify tag mapping is correct
    tag_id_map = cpp_config.tag_id_map
    assert len(tag_id_map) == 4  # alpha, beta, delta, gamma (sorted)

    # Verify tags are assigned correctly (alpha=0, beta=1, delta=2, gamma=3)
    assert tag_id_map[0] == "alpha"
    assert tag_id_map[1] == "beta"
    assert tag_id_map[2] == "delta"
    assert tag_id_map[3] == "gamma"


def test_empty_tags_allowed():
    """Test that agents with no tags work correctly."""
    game_config = GameConfig()

    # Create agents with no tags
    game_config.agents = [
        AgentConfig(team_id=0, tags=[]),
        AgentConfig(team_id=1, tags=[]),
    ]

    # Should succeed
    cpp_config = convert_to_cpp_game_config(game_config)

    # Verify no tags in mapping
    tag_id_map = cpp_config.tag_id_map
    assert len(tag_id_map) == 0


def test_default_agent_tags_preserved():
    """Test that default agent tags are preserved when agents list is empty."""
    # This test verifies the fix for the issue where default-agent tags were dropped
    # when agents list was empty but game_config.agent.tags was set

    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=2,  # Will create 2 default agents
            max_steps=100,
            obs_width=3,
            obs_height=3,
            num_observation_tokens=NUM_OBS_TOKENS,
            actions=ActionsConfig(noop=ActionConfig()),
            agent=AgentConfig(
                tags=["default_tag1", "default_tag2"]  # Tags for default agents
            ),
            agents=[],  # Empty agents list - will use defaults from agent field
            resource_names=[],
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    [".", "@", "."],
                    [".", ".", "."],
                    [".", "@", "."],
                ],
                char_to_name_map=DEFAULT_CHAR_TO_NAME,
            ),
        )
    )

    # Create environment - this will trigger convert_to_cpp_game_config
    env = MettaGridCore(cfg)
    obs, _ = env.reset()

    assert obs is not None
    assert len(obs) == 2  # Two default agents

    # Get tag feature ID from environment
    tag_feature_id = env.c_env.feature_spec()["tag"]["id"]

    # Check both agents have the default tags
    for agent_idx in range(2):
        agent_obs = obs[agent_idx]

        # Find agent locations
        agent_locations = set()
        for token in agent_obs:
            if token[1] == 0 and token[2] == 0:  # TypeId feature with agent value
                agent_locations.add(token[0])

        # Find tag IDs at agent locations
        agent_tag_ids = set()
        for token in agent_obs:
            if token[0] in agent_locations and token[1] == tag_feature_id:
                agent_tag_ids.add(token[2])

        # Each default agent should have 2 tags (default_tag1, default_tag2)
        assert len(agent_tag_ids) == 2, f"Default agent {agent_idx} should have 2 tags, found {len(agent_tag_ids)}"
        # Tag IDs should be 0 and 1 (alphabetically sorted: default_tag1=0, default_tag2=1)
        assert agent_tag_ids == {0, 1}, f"Default agent {agent_idx} should have tag IDs {{0, 1}}, got {agent_tag_ids}"


def test_default_agent_tags_in_cpp_config():
    """Test that default agent tags are included in the cpp config tag mapping."""
    game_config = GameConfig()

    # Set default agent tags but leave agents list empty
    game_config.agent = AgentConfig(tags=["hero", "player"])
    game_config.agents = []  # Empty - will create default agents
    game_config.num_agents = 3  # Will create 3 default agents

    # Convert to cpp config
    cpp_config = convert_to_cpp_game_config(game_config)

    # Verify tag mapping includes the default agent tags
    tag_id_map = cpp_config.tag_id_map
    assert len(tag_id_map) == 2, f"Should have 2 tags in mapping, got {len(tag_id_map)}"

    # Tags should be sorted alphabetically: hero=0, player=1
    assert tag_id_map[0] == "hero", f"Tag ID 0 should be 'hero', got {tag_id_map[0]}"
    assert tag_id_map[1] == "player", f"Tag ID 1 should be 'player', got {tag_id_map[1]}"


def test_tag_mapping_in_feature_spec():
    """Test that tag mapping is exposed through feature_spec()"""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=3,
            obs_height=3,
            num_observation_tokens=NUM_OBS_TOKENS,
            actions=ActionsConfig(noop=ActionConfig()),
            objects={
                "wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID, tags=["solid", "blocking"]),
                "converter": ConverterConfig(
                    type_id=2,
                    input_resources={"wood": 1},
                    output_resources={"coal": 1},
                    max_output=10,
                    max_conversions=5,
                    conversion_ticks=10,
                    cooldown=[5],
                    tags=["machine", "industrial"],
                ),
            },
            agents=[
                AgentConfig(tags=["player", "mobile"]),
            ],
            resource_names=["wood", "coal"],
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#"],
                    ["#", "@", "#"],
                    ["#", "#", "#"],
                ],
                char_to_name_map=DEFAULT_CHAR_TO_NAME,
            ),
        )
    )

    env = MettaGridCore(cfg)
    feature_spec = env.c_env.feature_spec()

    # Check that tag feature exists
    assert "tag" in feature_spec, "tag feature should be in feature_spec"

    tag_spec = feature_spec["tag"]

    # Check that tag feature has expected fields
    assert "id" in tag_spec, "tag feature should have 'id' field"
    assert "normalization" in tag_spec, "tag feature should have 'normalization' field"
    assert "values" in tag_spec, "tag feature should have 'values' field for tag mapping"

    # Check tag mapping contents
    tag_values = tag_spec["values"]
    assert isinstance(tag_values, dict), "tag values should be a dict mapping tag_id -> tag_name"

    # All unique tags sorted: ["blocking", "industrial", "machine", "mobile", "player", "solid"]
    # IDs should be 0-5
    expected_tags = ["blocking", "industrial", "machine", "mobile", "player", "solid"]
    assert len(tag_values) == len(expected_tags), f"Should have {len(expected_tags)} tags, got {len(tag_values)}"

    # Verify tags are sorted alphabetically with correct IDs
    for i, expected_tag in enumerate(expected_tags):
        assert i in tag_values, f"Tag ID {i} should be in mapping"
        assert tag_values[i] == expected_tag, f"Tag ID {i} should be '{expected_tag}', got '{tag_values[i]}'"


def test_tag_mapping_empty_tags():
    """Test that tag mapping works correctly when there are no tags."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=3,
            obs_height=3,
            num_observation_tokens=NUM_OBS_TOKENS,
            actions=ActionsConfig(noop=ActionConfig()),
            objects={
                "wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID, tags=[]),
            },
            agents=[
                AgentConfig(tags=[]),  # No tags
            ],
            resource_names=[],
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#"],
                    ["#", "@", "#"],
                    ["#", "#", "#"],
                ],
                char_to_name_map=DEFAULT_CHAR_TO_NAME,
            ),
        )
    )

    env = MettaGridCore(cfg)
    feature_spec = env.c_env.feature_spec()

    # Check that tag feature exists
    assert "tag" in feature_spec, "tag feature should be in feature_spec even with no tags"

    tag_spec = feature_spec["tag"]

    # Check that tag feature has expected fields
    assert "values" in tag_spec, "tag feature should have 'values' field even with no tags"

    # Check tag mapping is empty
    tag_values = tag_spec["values"]
    assert isinstance(tag_values, dict), "tag values should be a dict"
    assert len(tag_values) == 0, f"Should have 0 tags, got {len(tag_values)}"
