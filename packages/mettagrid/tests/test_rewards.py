import numpy as np

from mettagrid.config.mettagrid_c_config import from_mettagrid_config
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AttackActionConfig,
    ChangeGlyphActionConfig,
    ConverterConfig,
    GameConfig,
    WallConfig,
)
from mettagrid.mettagrid_c import (
    MettaGrid,
    dtype_actions,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)
from mettagrid.test_support import Orientation
from mettagrid.test_support.actions import (
    get_agent_position,
    move,
    noop,
)

NUM_AGENTS = 1
OBS_HEIGHT = 3
OBS_WIDTH = 3
NUM_OBS_TOKENS = 100
OBS_TOKEN_SIZE = 3


def create_heart_reward_test_env(max_steps=50, num_agents=NUM_AGENTS):
    """Helper function to create a MettaGrid environment with heart collection for reward testing."""

    # Create a simple map with agent, altar, and walls
    game_map = [
        ["wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "altar", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall"],
    ]

    game_config = GameConfig(
        max_steps=max_steps,
        num_agents=num_agents,
        obs_width=OBS_WIDTH,
        obs_height=OBS_HEIGHT,
        num_observation_tokens=NUM_OBS_TOKENS,
        resource_names=["laser", "armor", "heart"],
        actions=ActionsConfig(
            noop=ActionConfig(enabled=True),
            get_items=ActionConfig(enabled=True),
            move=ActionConfig(enabled=True),
            rotate=ActionConfig(enabled=True),
            put_items=ActionConfig(enabled=True),
            attack=AttackActionConfig(
                enabled=True,
                consumed_resources={"laser": 1},
                defense_resources={"armor": 1},
            ),
            swap=ActionConfig(enabled=True),
            change_glyph=ChangeGlyphActionConfig(enabled=False, number_of_glyphs=4),
        ),
        objects={
            "wall": WallConfig(type_id=1),
            "altar": ConverterConfig(
                type_id=8,
                output_resources={"heart": 1},
                initial_resource_count=5,
                max_output=50,
                conversion_ticks=1,
                cooldown=[10],
            ),
        },
        agent=AgentConfig(
            default_resource_limit=10,
            rewards=AgentRewards(
                inventory={"heart": 1.0},
            ),
        ),
    )

    return MettaGrid(from_mettagrid_config(game_config), game_map, 42)


def perform_action(env, action_name):
    """Perform a single action and return results."""
    available_actions = env.action_names()

    if action_name not in available_actions:
        raise ValueError(f"Unknown action '{action_name}'. Available actions: {available_actions}")

    action_idx = available_actions.index(action_name)
    action = np.full((NUM_AGENTS,), action_idx, dtype=dtype_actions)
    obs, rewards, _terminals, _truncations, _info = env.step(action)
    return obs, float(rewards[0]), env.action_success()[0]


def wait_for_heart_production(env, steps=5):
    """Wait for altar to produce hearts by performing noop actions."""
    for _ in range(steps):
        noop(env)


def collect_heart_from_altar(env):
    """Move agent to altar (if needed) and collect a heart. Returns (success, reward)."""
    agent_pos = get_agent_position(env, 0)
    # Agent starts at (1, 1), altar is at (1, 3)
    # Position (1, 2) is directly left of altar
    target_pos = (1, 2)

    # Move to target position if not already there
    if agent_pos != target_pos:
        # Agent starts at (1, 1), needs to move right to (1, 2)
        move_result = move(env, Orientation.EAST, agent_idx=0)
        if not move_result["success"]:
            return False, 0.0

    # Collect heart from adjacent altar
    _obs, reward, success = perform_action(env, "get_items")
    return success, reward


class TestRewards:
    def test_step_rewards_initialization(self):
        """Test that step rewards are properly initialized to zero."""
        env = create_heart_reward_test_env()

        # Create buffers
        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=dtype_observations)
        terminals = np.zeros(NUM_AGENTS, dtype=dtype_terminals)
        truncations = np.zeros(NUM_AGENTS, dtype=dtype_truncations)
        rewards = np.zeros(NUM_AGENTS, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # Check that rewards start at zero
        assert np.all(rewards == 0), f"Rewards should start at zero, got {rewards}"

        # Take a step with noop action
        noop_result = noop(env)
        assert noop_result["success"], "Noop should always succeed"

        # Get the rewards from the step
        step_rewards = rewards.copy()  # The buffer is updated by step()

        # Check that step rewards are accessible and match buffer
        assert np.array_equal(step_rewards, rewards), "Step rewards should match buffer rewards"
        print(f"✅ Step rewards properly initialized: {step_rewards}")

    def test_heart_collection_rewards(self):
        """Test that collecting hearts generates real rewards."""
        env = create_heart_reward_test_env()

        # Create buffers
        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=dtype_observations)
        terminals = np.zeros(NUM_AGENTS, dtype=dtype_terminals)
        truncations = np.zeros(NUM_AGENTS, dtype=dtype_truncations)
        rewards = np.zeros(NUM_AGENTS, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # Collect heart and verify rewards
        success, reward = collect_heart_from_altar(env)

        assert success, "Heart collection should succeed"
        assert reward > 0, f"Heart collection should give positive reward, got {reward}"

        # Check episode rewards
        episode_rewards = env.get_episode_rewards()
        assert episode_rewards[0] > 0, f"Episode rewards should be positive, got {episode_rewards[0]}"

        print(f"✅ Heart collection successful! Reward: {reward}, Episode total: {episode_rewards[0]}")

    def test_multiple_heart_collections(self):
        """Test collecting multiple hearts and verifying cumulative rewards."""
        env = create_heart_reward_test_env()

        # Create buffers
        observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=dtype_observations)
        terminals = np.zeros(NUM_AGENTS, dtype=dtype_terminals)
        truncations = np.zeros(NUM_AGENTS, dtype=dtype_truncations)
        rewards = np.zeros(NUM_AGENTS, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # First collection
        success1, reward1 = collect_heart_from_altar(env)
        episode_rewards_1 = env.get_episode_rewards()[0]

        # Wait and collect again
        wait_for_heart_production(env, steps=10)
        success2, reward2 = collect_heart_from_altar(env)
        episode_rewards_2 = env.get_episode_rewards()[0]

        # Verify both collections worked
        assert success1, "First collection should succeed"
        assert success2, "Second collection should succeed"
        assert reward1 > 0, f"First collection should give positive reward, got {reward1}"
        assert reward2 > 0, f"Second collection should give positive reward, got {reward2}"

        # Verify episode rewards accumulate
        assert episode_rewards_2 > episode_rewards_1, "Episode rewards should accumulate"
        expected_total = episode_rewards_1 + reward2
        assert abs(episode_rewards_2 - expected_total) < 1e-6, (
            f"Episode rewards should accumulate correctly: {episode_rewards_2} vs {expected_total}"
        )

        print("✅ Multiple collections successful!")
        print(f"   Collection 1: reward={reward1}, episode_total={episode_rewards_1}")
        print(f"   Collection 2: reward={reward2}, episode_total={episode_rewards_2}")
