import numpy as np
import pytest

from mettagrid.config.mettagrid_c_config import from_mettagrid_config
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AttackActionConfig,
    ChangeGlyphActionConfig,
    GameConfig,
    WallConfig,
)
from mettagrid.map_builder.utils import create_grid
from mettagrid.mettagrid_c import (
    MettaGrid,
    dtype_actions,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)

NUM_AGENTS = 2
OBS_HEIGHT = 3
OBS_WIDTH = 3
NUM_OBS_TOKENS = 30
OBS_TOKEN_SIZE = 3


def create_minimal_mettagrid_c_env(max_steps=10, width=5, height=5, config_overrides: dict | None = None):
    """Helper function to create a MettaGrid environment with minimal config.

    Args:
        max_steps: Maximum steps before truncation
        width: Map width
        height: Map height
        config_overrides: Dictionary to override GameConfig fields
    """
    # Define a simple map: empty with walls around perimeter
    game_map = create_grid(height, width)
    game_map[0, :] = "wall"
    game_map[-1, :] = "wall"
    game_map[:, 0] = "wall"
    game_map[:, -1] = "wall"
    # Place first agent in upper left
    game_map[1, 1] = "agent.red"
    # Place second agent in middle
    mid_y = height // 2
    mid_x = width // 2
    game_map[mid_y, mid_x] = "agent.red"

    game_config = GameConfig(
        max_steps=max_steps,
        num_agents=NUM_AGENTS,
        obs_width=OBS_WIDTH,
        obs_height=OBS_HEIGHT,
        num_observation_tokens=NUM_OBS_TOKENS,
        resource_names=["laser", "armor"],
        actions=ActionsConfig(
            noop=ActionConfig(enabled=True),
            move=ActionConfig(enabled=True),
            attack=AttackActionConfig(enabled=False),
            put_items=ActionConfig(enabled=False),
            get_items=ActionConfig(enabled=False),
            swap=ActionConfig(enabled=False),
            change_glyph=ChangeGlyphActionConfig(enabled=True, number_of_glyphs=4),
        ),
        objects={"wall": WallConfig(type_id=1)},
        agent=AgentConfig(),
    )

    # Apply config overrides if provided
    if config_overrides:
        game_config = game_config.model_copy(update=config_overrides)

    return MettaGrid(from_mettagrid_config(game_config), game_map.tolist(), 42)


class TestBuffers:
    """Comprehensive tests for MettaGrid buffer functionality."""

    def test_default_buffers_in_gym_mode(self):
        """Test that buffers work correctly in gym mode (without explicit set_buffers call)."""
        c_env = create_minimal_mettagrid_c_env()
        c_env.reset()

        noop_action_idx = c_env.action_names().index("noop")
        actions = np.full(NUM_AGENTS, noop_action_idx, dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = c_env.step(actions)
        episode_rewards = c_env.get_episode_rewards()

        # Check strides. We've had issues where we've not correctly initialized the buffers, and have had
        # strides of zero.
        assert rewards.strides == (4,)  # float32
        assert terminals.strides == (1,)  # bool, tracked as a byte
        assert truncations.strides == (1,)  # bool, tracked as a byte
        assert episode_rewards.strides == (4,)  # float32
        assert obs.strides[-1] == 1  # uint8

        # This is a more brute force way to check that the buffers are behaving correctly by changing a single
        # element and making sure the correct update is reflected. Given that the strides are correct, these tests
        # are probably superfluous; but we've been surprised by what can fail in the past, so we're aiming for
        # overkill.
        assert (rewards == [0, 0]).all()
        assert (terminals == [False, False]).all()
        assert (truncations == [False, False]).all()
        assert (episode_rewards == [0, 0]).all()

        rewards[0] = 1
        terminals[0] = True
        truncations[0] = True
        episode_rewards[0] = 1

        assert (rewards == [1, 0]).all()
        assert (terminals == [True, False]).all()
        assert (truncations == [True, False]).all()
        assert (episode_rewards == [1, 0]).all()

        # Obs is non-empty, so we treat it differently than the others.
        initial_obs_sum = obs.sum()
        obs[0, 0, 0] += 1
        assert obs.sum() == initial_obs_sum + 1

    def test_set_buffers_wrong_shape(self):
        """Test that set_buffers properly validates buffer shapes."""
        c_env = create_minimal_mettagrid_c_env()
        terminals = np.zeros(NUM_AGENTS, dtype=bool)
        truncations = np.zeros(NUM_AGENTS, dtype=bool)
        rewards = np.zeros(NUM_AGENTS, dtype=np.float32)

        # Wrong number of agents
        observations = np.zeros((3, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="observations"):
            c_env.set_buffers(observations, terminals, truncations, rewards)

        # Wrong token size
        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE - 1), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="observations"):
            c_env.set_buffers(observations, terminals, truncations, rewards)

        # Wrong number of agents for other buffers
        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=np.uint8)
        wrong_terminals = np.zeros(NUM_AGENTS + 1, dtype=bool)
        with pytest.raises(RuntimeError):
            c_env.set_buffers(observations, wrong_terminals, truncations, rewards)

        wrong_truncations = np.zeros(NUM_AGENTS - 1, dtype=bool)
        with pytest.raises(RuntimeError):
            c_env.set_buffers(observations, terminals, wrong_truncations, rewards)

        wrong_rewards = np.zeros(NUM_AGENTS + 2, dtype=np.float32)
        with pytest.raises(RuntimeError):
            c_env.set_buffers(observations, terminals, truncations, wrong_rewards)

    def test_set_buffers_wrong_dtype(self):
        """Test that set_buffers properly validates buffer dtypes."""
        c_env = create_minimal_mettagrid_c_env()

        # Correct buffers for comparison
        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=dtype_observations)
        terminals = np.zeros(NUM_AGENTS, dtype=dtype_terminals)
        truncations = np.zeros(NUM_AGENTS, dtype=dtype_truncations)
        rewards = np.zeros(NUM_AGENTS, dtype=dtype_rewards)

        # Wrong observation dtype
        wrong_obs = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=np.float32)
        assert wrong_obs.dtype != dtype_observations
        with pytest.raises(TypeError):
            c_env.set_buffers(wrong_obs, terminals, truncations, rewards)

        # Wrong terminals dtype
        wrong_terminals = np.zeros(NUM_AGENTS, dtype=np.int32)
        assert wrong_terminals.dtype != dtype_terminals
        with pytest.raises(TypeError):
            c_env.set_buffers(observations, wrong_terminals, truncations, rewards)

        # Wrong truncations dtype
        wrong_truncations = np.zeros(NUM_AGENTS, dtype=np.int32)
        assert wrong_truncations.dtype != dtype_truncations
        with pytest.raises(TypeError):
            c_env.set_buffers(observations, terminals, wrong_truncations, rewards)

        # Wrong rewards dtype
        wrong_rewards = np.zeros(NUM_AGENTS, dtype=np.float64)
        assert wrong_rewards.dtype != dtype_rewards
        with pytest.raises(TypeError):
            c_env.set_buffers(observations, terminals, truncations, wrong_rewards)

    def test_set_buffers_non_contiguous(self):
        """Test that set_buffers requires C-contiguous arrays."""
        c_env = create_minimal_mettagrid_c_env()

        # Create non-contiguous arrays
        observations = np.asfortranarray(np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=np.uint8))
        terminals = np.zeros(NUM_AGENTS, dtype=bool)
        truncations = np.zeros(NUM_AGENTS, dtype=bool)
        rewards = np.zeros(NUM_AGENTS, dtype=np.float32)

        with pytest.raises(TypeError):
            c_env.set_buffers(observations, terminals, truncations, rewards)

        # Test with other non-contiguous buffers
        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=np.uint8)

        temp = np.zeros((NUM_AGENTS * 2,), dtype=bool)
        non_contiguous_terminals = temp[::2][:NUM_AGENTS]
        with pytest.raises(TypeError):
            c_env.set_buffers(observations, non_contiguous_terminals, truncations, rewards)

    def test_set_buffers_happy_path(self):
        """Test successful buffer setup and basic functionality."""
        c_env = create_minimal_mettagrid_c_env()
        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=np.uint8)
        terminals = np.zeros(NUM_AGENTS, dtype=bool)
        truncations = np.zeros(NUM_AGENTS, dtype=bool)
        rewards = np.zeros(NUM_AGENTS, dtype=np.float32)

        c_env.set_buffers(observations, terminals, truncations, rewards)
        observations_from_env, info = c_env.reset()
        np.testing.assert_array_equal(observations_from_env, observations)

    def test_buffer_memory_sharing_and_overwriting(self):
        """Test that all buffers share memory with environment and are properly overwritten during steps."""
        c_env = create_minimal_mettagrid_c_env()

        # Create buffers
        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=np.uint8)
        terminals = np.zeros(NUM_AGENTS, dtype=bool)
        truncations = np.zeros(NUM_AGENTS, dtype=bool)
        rewards = np.zeros(NUM_AGENTS, dtype=np.float32)

        # Verify buffer properties
        assert observations.flags.c_contiguous, "Observations buffer should be C-contiguous"
        assert terminals.flags.c_contiguous, "Terminals buffer should be C-contiguous"
        assert truncations.flags.c_contiguous, "Truncations buffer should be C-contiguous"
        assert rewards.flags.c_contiguous, "Rewards buffer should be C-contiguous"

        assert observations.dtype == np.uint8, "Observations should be uint8"
        assert terminals.dtype == bool, "Terminals should be bool"
        assert truncations.dtype == bool, "Truncations should be bool"
        assert rewards.dtype == np.float32, "Rewards should be float32"

        c_env.set_buffers(observations, terminals, truncations, rewards)
        c_env.reset()

        # Manually set values in all buffers to test memory sharing
        observations[0, 0, 0] = 255
        observations[1, 1, 1] = 128
        terminals[0] = True
        terminals[1] = False
        truncations[0] = False
        truncations[1] = True
        rewards[0] = 99.5
        rewards[1] = -42.3

        # Take a step - this should overwrite our manual values
        noop_action_idx = c_env.action_names().index("noop")
        actions = np.full(NUM_AGENTS, noop_action_idx, dtype=dtype_actions)

        obs_returned, rewards_returned, terminals_returned, truncations_returned, info = c_env.step(actions)

        # Verify that step overwrote our manual values for actively managed buffers
        # (observations will be overwritten with actual game state)
        assert not (observations[0, 0, 0] == 255 and observations[1, 1, 1] == 128), (
            "Step should have overwritten manual observation values"
        )
        assert not np.array_equal(rewards, [99.5, -42.3]), "Step should have overwritten manual reward values"

        # NOTE: Terminals are not actively managed by this environment - they remain unchanged
        # This is intentional behavior as the environment doesn't use terminal states

        # NOTE: Truncations are only written when max_steps is reached, not on every step
        # Since this is a single step and we haven't reached max_steps, truncations remain unchanged
        # This is intentional behavior

        assert np.array_equal(observations, obs_returned), "Observations buffer should share memory"
        assert np.array_equal(rewards, rewards_returned), "Rewards buffer should share memory"
        assert np.array_equal(terminals, terminals_returned), "Terminals buffer should share memory"
        assert np.array_equal(truncations, truncations_returned), "Truncations buffer should share memory"

    def test_truncations_on_max_steps(self):
        """Test that truncations are set when max_steps is reached."""
        # Create environment with max_steps = 1
        c_env = create_minimal_mettagrid_c_env(config_overrides={"max_steps": 1, "episode_truncates": True})

        # Set up buffers
        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=np.uint8)
        terminals = np.zeros(NUM_AGENTS, dtype=bool)
        truncations = np.zeros(NUM_AGENTS, dtype=bool)
        rewards = np.zeros(NUM_AGENTS, dtype=np.float32)

        c_env.set_buffers(observations, terminals, truncations, rewards)
        c_env.reset()  # current_step = 0

        # Take one step to reach max_steps
        noop_action_idx = c_env.action_names().index("noop")
        actions = np.full(NUM_AGENTS, noop_action_idx, dtype=dtype_actions)
        c_env.step(actions)  # current_step = 1, should trigger end of episode

        # Now truncations should all be True
        assert np.all(truncations), "All agents should be truncated when max_steps is reached"
        assert not np.any(terminals), "All agents should not be terminated when max_steps is reached"

    def test_terminals_on_max_steps(self):
        """Test that truncations are set when max_steps is reached."""
        # Create environment with max_steps = 1
        c_env = create_minimal_mettagrid_c_env(config_overrides={"max_steps": 1})

        # Set up buffers
        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=np.uint8)
        terminals = np.zeros(NUM_AGENTS, dtype=bool)
        truncations = np.zeros(NUM_AGENTS, dtype=bool)
        rewards = np.zeros(NUM_AGENTS, dtype=np.float32)

        c_env.set_buffers(observations, terminals, truncations, rewards)
        c_env.reset()  # current_step = 0

        # Take one step to reach max_steps
        noop_action_idx = c_env.action_names().index("noop")
        actions = np.full(NUM_AGENTS, noop_action_idx, dtype=dtype_actions)
        c_env.step(actions)  # current_step = 1, should trigger end of episode

        # Now terminals should all be True, truncations should all be False
        assert np.all(terminals), "All agents should be terminated when max_steps is reached"
        assert not np.any(truncations), "All agents should not be truncated when max_steps is reached"

    def test_buffer_element_modification_independence(self):
        """Test that modifying individual buffer elements works correctly across all buffer types."""
        c_env = create_minimal_mettagrid_c_env()

        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=np.uint8)
        terminals = np.zeros(NUM_AGENTS, dtype=bool)
        truncations = np.zeros(NUM_AGENTS, dtype=bool)
        rewards = np.zeros(NUM_AGENTS, dtype=np.float32)

        c_env.set_buffers(observations, terminals, truncations, rewards)
        c_env.reset()

        # Take a step to get valid baseline values
        noop_action_idx = c_env.action_names().index("noop")
        actions = np.full(NUM_AGENTS, noop_action_idx, dtype=dtype_actions)
        c_env.step(actions)

        # Store initial values
        initial_obs_sum = observations.sum()
        initial_terminals = terminals.copy()
        initial_truncations = truncations.copy()
        initial_rewards = rewards.copy()

        # Modify individual elements in each buffer
        observations[0, 0, 0] += 1
        terminals[0] = not terminals[0]  # Flip the boolean
        truncations[1] = not truncations[1]  # Flip the boolean
        rewards[0] += 10.0

        # Verify modifications affected only the intended elements
        assert observations.sum() == initial_obs_sum + 1, "Observation modification should affect only one element"

        expected_terminals = initial_terminals.copy()
        expected_terminals[0] = not initial_terminals[0]
        np.testing.assert_array_equal(terminals, expected_terminals, "Terminal modification should affect only agent 0")

        expected_truncations = initial_truncations.copy()
        expected_truncations[1] = not initial_truncations[1]
        np.testing.assert_array_equal(
            truncations, expected_truncations, "Truncation modification should affect only agent 1"
        )

        expected_rewards = initial_rewards.copy()
        expected_rewards[0] += 10.0
        np.testing.assert_array_equal(rewards, expected_rewards, "Reward modification should affect only agent 0")

    def test_multi_agent_buffer_behavior(self):
        """Test buffer behavior with multiple agents to ensure proper indexing."""
        c_env = create_minimal_mettagrid_c_env()

        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=np.uint8)
        terminals = np.zeros(NUM_AGENTS, dtype=bool)
        truncations = np.zeros(NUM_AGENTS, dtype=bool)
        rewards = np.zeros(NUM_AGENTS, dtype=np.float32)

        c_env.set_buffers(observations, terminals, truncations, rewards)
        c_env.reset()

        # Verify all agents have independent buffer space
        for agent_idx in range(NUM_AGENTS):
            # Each agent should have its own observation space
            assert observations[agent_idx].shape == (NUM_OBS_TOKENS, OBS_TOKEN_SIZE), (
                f"Agent {agent_idx} should have correct observation shape"
            )

            # Each agent should have independent scalar values
            original_terminal = terminals[agent_idx]
            original_truncation = truncations[agent_idx]
            original_reward = rewards[agent_idx]

            # Modify this agent's values
            terminals[agent_idx] = not original_terminal
            truncations[agent_idx] = not original_truncation
            rewards[agent_idx] = float(agent_idx + 100)

            # Verify other agents weren't affected
            for other_agent in range(NUM_AGENTS):
                if other_agent != agent_idx:
                    assert (
                        terminals[other_agent] != terminals[agent_idx] or original_terminal == terminals[agent_idx]
                    ), f"Agent {other_agent} terminals should not be affected by agent {agent_idx} modification"

            # Reset for next iteration
            terminals[agent_idx] = original_terminal
            truncations[agent_idx] = original_truncation
            rewards[agent_idx] = original_reward

    def test_episode_rewards_accumulation(self):
        """Test that episode rewards properly accumulate across steps with custom buffers."""
        c_env = create_minimal_mettagrid_c_env()

        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=np.uint8)
        terminals = np.zeros(NUM_AGENTS, dtype=bool)
        truncations = np.zeros(NUM_AGENTS, dtype=bool)
        rewards = np.zeros(NUM_AGENTS, dtype=np.float32)

        c_env.set_buffers(observations, terminals, truncations, rewards)
        c_env.reset()

        # Get initial episode rewards - should be zero
        episode_rewards = c_env.get_episode_rewards()
        assert np.all(episode_rewards == 0), f"Episode rewards should start at zero, got {episode_rewards}"

        # Take first step
        noop_action_idx = c_env.action_names().index("noop")
        actions = np.full(NUM_AGENTS, noop_action_idx, dtype=dtype_actions)

        obs, step_rewards_1, terminals_ret, truncations_ret, _info = c_env.step(actions)
        episode_rewards_1 = c_env.get_episode_rewards()

        # Episode rewards should equal step rewards after first step
        np.testing.assert_array_equal(
            episode_rewards_1, step_rewards_1, "Episode rewards should equal step rewards after first step"
        )

        # Verify buffers match returned values
        np.testing.assert_array_equal(step_rewards_1, rewards, "Step rewards should match buffer")
        np.testing.assert_array_equal(terminals_ret, terminals, "Terminals should match buffer")
        np.testing.assert_array_equal(truncations_ret, truncations, "Truncations should match buffer")

        # Take second step
        obs, step_rewards_2, terminals_ret, truncations_ret, _info = c_env.step(actions)
        episode_rewards_2 = c_env.get_episode_rewards()

        # Episode rewards should be cumulative
        expected_cumulative = episode_rewards_1 + step_rewards_2
        np.testing.assert_array_equal(episode_rewards_2, expected_cumulative, "Episode rewards should be cumulative")
