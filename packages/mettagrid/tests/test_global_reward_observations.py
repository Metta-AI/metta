import pytest

# this comment is here to help with a ruff linting bug
from mettagrid.mettagrid_c import (
    AgentConfig,
    GameConfig,
    GlobalObsConfig,
    MettaGrid,
)


@pytest.fixture
def resource_names():
    """Standard inventory items for testing."""
    return [f"item{i}" for i in range(10)]


@pytest.fixture
def base_agent_config():
    """Factory for creating base agent configurations."""

    def _create(resource_names, resource_rewards=None):
        return AgentConfig(
            type_id=1,
            type_name="agent",
            group_id=0,
            group_name="test_group",
            freeze_duration=0,
            action_failure_penalty=0.0,
            resource_limits={i: 100 for i in range(len(resource_names))},
            resource_rewards=resource_rewards or {i: 0.0 for i in range(len(resource_names))},
            resource_reward_max={i: 100 for i in range(len(resource_names))},
            group_reward_pct=0.0,
        )

    return _create


@pytest.fixture
def global_obs_config():
    """Global observation config with resource rewards enabled."""
    return GlobalObsConfig(resource_rewards=True)


@pytest.fixture
def game_config_factory(global_obs_config):
    """Factory for creating game configurations."""

    def _create(num_agents, resource_names, agent_config):
        return GameConfig(
            num_agents=num_agents,
            max_steps=100,
            episode_truncates=False,
            obs_width=5,
            obs_height=5,
            resource_names=resource_names,
            num_observation_tokens=50,
            global_obs=global_obs_config,
            actions=[],
            objects={"agent.test_group": agent_config},
            track_movement_metrics=False,
        )

    return _create


@pytest.fixture
def create_env(game_config_factory):
    """Factory for creating MettaGrid environments."""

    def _create(num_agents, resource_names, agent_config, seed=42):
        game_config = game_config_factory(num_agents, resource_names, agent_config)
        game_map = [["agent.test_group"] * num_agents]
        return MettaGrid(game_config, game_map, seed)

    return _create


@pytest.fixture
def find_resource_rewards_token():
    """Helper to find resource rewards token in observations."""

    def _find(feature_spec, agent_obs):
        feature_id = feature_spec["resource_rewards"]["id"]
        for token in agent_obs:
            if token[0] != 0xFF and token[1] == feature_id:
                return token
        return None

    return _find


class TestGlobalRewardObservations:
    """Test global reward observation features in MettaGrid."""

    def test_resource_rewards_observation(
        self,
        resource_names,
        base_agent_config,
        create_env,
        find_resource_rewards_token,
    ):
        """Test that inventory rewards global observation is correctly packed and included."""
        # Create agent config with specific reward values
        resource_rewards = {
            0: 0.25,  # item0: has reward (bit = 1)
            1: 0.0,  # item1: no reward (bit = 0)
            2: 1.5,  # item2: has reward (bit = 1)
            3: 0.0,  # item3: no reward (bit = 0)
            4: 2.0,  # item4: has reward (bit = 1)
            5: -0.5,  # item5: negative reward (bit = 0)
            6: 0.1,  # item6: has reward (bit = 1)
            7: 0.0,  # item7: no reward (bit = 0)
            8: 5.0,  # item8: has reward but not included (beyond 8 items)
            9: 1.0,  # item9: has reward but not included (beyond 8 items)
        }
        agent_config = base_agent_config(resource_names, resource_rewards)

        # Create environment and get observations
        env = create_env(1, resource_names, agent_config)
        observations, _ = env.reset()

        # Check observations shape (1 agent, 50 tokens, 3 values per token)
        assert observations.shape == (1, 50, 3), f"Unexpected shape: {observations.shape}"

        # Find and verify resource rewards token
        resource_rewards_token = find_resource_rewards_token(env.feature_spec(), observations[0])
        assert resource_rewards_token is not None, "Inventory rewards token not found in observation"

        # Expected packed value: 10101010 = 0xAA = 170
        expected_packed = 0b10101010
        assert resource_rewards_token[2] == expected_packed, (
            f"Incorrect packed inventory rewards value. Expected: {expected_packed}, Got: {resource_rewards_token[2]}"
        )

    def test_resource_rewards_only_for_observing_agent(self, base_agent_config, create_env):
        """Test that inventory rewards observation is included as global token for each agent."""
        resource_names = ["item0", "item1", "item2", "item3"]
        resource_rewards = {i: 1.0 for i in range(len(resource_names))}
        agent_config = base_agent_config(resource_names, resource_rewards)

        # Create environment with two agents
        env = create_env(2, resource_names, agent_config)
        observations, _ = env.reset()

        feature_id = env.feature_spec()["resource_rewards"]["id"]

        # Check that each agent sees their own game rewards at center
        for agent_idx in range(2):
            agent_obs = observations[agent_idx]

            # Find game rewards token at center (position 2,2 for 5x5 observation)
            center_packed = (2 << 4) | 2  # Row 2, Col 2

            resource_rewards_found_at_center = False
            for token in agent_obs:
                if token[0] == center_packed and token[1] == feature_id:
                    resource_rewards_found_at_center = True
                    # All 4 items have rewards, so packed value should be 0b11110000 = 240
                    assert token[2] == 0b11110000, f"Wrong packed value for agent {agent_idx}: {token[2]}"
                    break

            assert resource_rewards_found_at_center, f"Game rewards not found at center for agent {agent_idx}"

    def test_resource_rewards_with_partial_items(self, base_agent_config, create_env, find_resource_rewards_token):
        """Test inventory rewards with fewer than 8 items."""
        resource_names = ["item0", "item1", "item2"]
        resource_rewards = {
            0: 2.0,  # item0: has reward (bit = 1)
            1: 0.0,  # item1: no reward (bit = 0)
            2: 0.3,  # item2: has reward (bit = 1)
        }
        agent_config = base_agent_config(resource_names, resource_rewards)

        # Create environment and get observations
        env = create_env(1, resource_names, agent_config)
        observations, _ = env.reset()

        # Find and verify resource rewards token
        resource_rewards_token = find_resource_rewards_token(env.feature_spec(), observations[0])
        assert resource_rewards_token is not None

        # Expected: 10100000 = 0xA0 = 160
        expected_packed = 0b10100000
        assert resource_rewards_token[2] == expected_packed

    @pytest.mark.parametrize(
        "rewards,expected_packed",
        [
            (
                {
                    0: 0.0,  # exactly 0 -> 0
                    1: 0.01,  # small positive -> 1
                    2: 1.0,  # positive -> 1
                    3: -1.0,  # negative -> 0
                    4: 100.0,  # large positive -> 1
                    5: -0.1,  # small negative -> 0
                    6: 0.5,  # positive -> 1
                    7: 0.0,  # zero -> 0
                },
                0b01101010,  # Expected: 01101010 = 0x6A = 106
            ),
            (
                {i: 0.0 for i in range(8)},  # All zeros
                0b00000000,
            ),
            (
                {i: 1.0 for i in range(8)},  # All positive
                0b11111111,
            ),
            (
                {i: -1.0 for i in range(8)},  # All negative
                0b00000000,
            ),
        ],
    )
    def test_resource_rewards_binary_quantization(
        self,
        base_agent_config,
        create_env,
        find_resource_rewards_token,
        rewards,
        expected_packed,
    ):
        """Test the binary quantization of reward values."""
        resource_names = [f"item{i}" for i in range(8)]
        agent_config = base_agent_config(resource_names, rewards)

        # Create environment and get observations
        env = create_env(1, resource_names, agent_config)
        observations, _ = env.reset()

        # Find and verify resource rewards token
        resource_rewards_token = find_resource_rewards_token(env.feature_spec(), observations[0])
        assert resource_rewards_token is not None
        assert resource_rewards_token[2] == expected_packed
