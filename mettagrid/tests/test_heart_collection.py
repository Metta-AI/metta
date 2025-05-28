import numpy as np
import pytest

from mettagrid.mettagrid_c import MettaGrid
from tests.actions import (
    Orientation,
    get_agent_position,
    move,
    np_actions_type,
    np_observations_type,
    np_rewards_type,
    np_terminals_type,
    np_truncations_type,
    rotate,
)

OBS_WIDTH = 3  # should be odd
OBS_HEIGHT = 3  # should be odd


@pytest.fixture
def heart_config():
    """Configuration for heart collection tests."""
    return {
        "max_steps": 50,
        "num_agents": 1,
        "obs_width": OBS_WIDTH,
        "obs_height": OBS_HEIGHT,
        "actions": {
            "noop": {"enabled": True},
            "get_items": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "put_items": {"enabled": True},
            "attack": {"enabled": True},
            "swap": {"enabled": True},
            "change_color": {"enabled": True},
        },
        "groups": {"red": {"id": 0, "props": {}}},
        "objects": {
            "wall": {"type_id": 1, "hp": 100},
            "altar": {
                "type_id": 4,
                "hp": 100,
                "output_heart": 1,
                "initial_items": 0,
                "max_output": 50,
                "conversion_ticks": 0,
                "cooldown": 100,
            },
        },
        "agent": {
            "default_item_max": 10,
            "hp": 100,
            "rewards": {"heart": 1.0},
        },
    }


@pytest.fixture
def heart_game_map():
    """Game map with agent and altar for heart collection."""
    return [
        ["wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "altar", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall"],
    ]


@pytest.fixture
def heart_env(heart_config):
    """Factory fixture that creates a heart collection environment."""

    def _create_env(game_map, config_overrides=None):
        config = heart_config.copy()
        if config_overrides:
            config.update(config_overrides)

        env_config = {"game": config}
        env = MettaGrid(env_config, game_map)

        num_features = len(env.grid_features())
        observations = np.zeros((1, 3, 3, num_features), dtype=np_observations_type)
        terminals = np.zeros(1, dtype=np_terminals_type)
        truncations = np.zeros(1, dtype=np_truncations_type)
        rewards = np.zeros(1, dtype=np_rewards_type)
        env.set_buffers(observations, terminals, truncations, rewards)

        obs, info = env.reset()
        return env, obs

    return _create_env


@pytest.fixture
def heart_helpers():
    """Helper functions for heart collection tests."""

    def get_agent_hearts(env, obs):
        """Get the number of hearts in agent's inventory from observations."""
        grid_features = env.grid_features()

        if "inv:heart" in grid_features:
            feature_idx = grid_features.index("inv:heart")
            return obs[OBS_WIDTH // 2, OBS_HEIGHT // 2, feature_idx]
        return 0

    def perform_action(env, action_name, arg=0):
        """Perform a single action and return results."""
        available_actions = env.action_names()

        if action_name not in available_actions:
            raise ValueError(f"Unknown action '{action_name}'. Available actions: {available_actions}")

        action_idx = available_actions.index(action_name)
        action = np.zeros((1, 2), dtype=np_actions_type)
        action[0] = [action_idx, arg]
        obs, rewards, _terminals, _truncations, _info = env.step(action)
        return obs, float(rewards[0]), env.action_success()[0]

    def get_altar_info(env):
        """Get altar position and heart count."""
        grid_objects = env.grid_objects()
        for _obj_id, obj_data in grid_objects.items():
            if "altar" in obj_data:
                altar_pos = (obj_data["r"], obj_data["c"])
                altar_hearts = obj_data.get("inv:heart", 0)
                return altar_pos, altar_hearts
        return None, 0

    def wait_for_heart_production(env, steps=5):
        """Wait for altar to produce hearts by performing noop actions."""
        for _ in range(steps):
            perform_action(env, "noop")

    def are_adjacent(pos1, pos2):
        """Check if two positions are adjacent (Manhattan distance = 1)."""
        if pos1 is None or pos2 is None:
            return False
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    return {
        "get_agent_hearts": get_agent_hearts,
        "perform_action": perform_action,
        "get_altar_info": get_altar_info,
        "wait_for_heart_production": wait_for_heart_production,
        "are_adjacent": are_adjacent,
    }


def test_heart_collection_basic(heart_env, heart_game_map, heart_helpers):
    """Test basic heart collection from altar."""
    env, obs = heart_env(heart_game_map)
    helpers = heart_helpers

    # Get initial positions
    agent_pos = get_agent_position(env, 0)
    altar_pos, _ = helpers["get_altar_info"](env)

    assert agent_pos is not None, "Agent should have a valid position"
    assert altar_pos is not None, "Altar should be found on the map"

    print(f"Agent at {agent_pos}, Altar at {altar_pos}")

    # Wait for heart production
    helpers["wait_for_heart_production"](env, steps=5)

    # Move agent to be adjacent to altar (move right from starting position)
    move_result = move(env, Orientation.RIGHT)
    assert move_result["success"], f"Movement should succeed. Error: {move_result.get('error')}"

    # Verify positions after movement
    agent_pos = get_agent_position(env, 0)
    altar_pos, altar_hearts = helpers["get_altar_info"](env)

    assert helpers["are_adjacent"](agent_pos, altar_pos), (
        f"Agent at {agent_pos} should be adjacent to altar at {altar_pos}"
    )

    print(f"After movement: Agent at {agent_pos}, Altar at {altar_pos} with {altar_hearts} hearts")

    hearts_before = helpers["get_agent_hearts"](env, obs[0])

    # Rotate to face the altar
    rotate_result = rotate(env, Orientation.RIGHT)
    assert rotate_result["success"], f"Rotation to {Orientation.RIGHT} should succeed"

    # Try to collect hearts
    obs, reward, success = helpers["perform_action"](env, "get_output", 0)
    hearts_after = helpers["get_agent_hearts"](env, obs[0])
    hearts_gained = hearts_after - hearts_before

    print(f"Facing {Orientation.RIGHT}: hearts {hearts_before} → {hearts_after}, reward: {reward}, success: {success}")

    hearts_collected = success and hearts_gained > 0

    if hearts_collected:
        print(f"✅ Successfully collected {hearts_gained} heart(s) when facing {Orientation.RIGHT}")

    assert hearts_collected, "Should be able to collect hearts from altar in at least one orientation"


def test_heart_collection_with_reward(heart_env, heart_game_map, heart_helpers):
    """Test that collecting hearts gives positive reward."""
    env, obs = heart_env(heart_game_map)
    helpers = heart_helpers

    # Setup: move to altar and wait for heart production
    helpers["wait_for_heart_production"](env, steps=5)
    move_result = move(env, Orientation.RIGHT)
    assert move_result["success"], "Should be able to move to altar"

    # Find the correct orientation for collection
    collection_reward = 0
    for orientation in [Orientation.UP, Orientation.DOWN, Orientation.LEFT, Orientation.RIGHT]:
        rotate_result = rotate(env, orientation)
        assert rotate_result["success"], f"Should be able to rotate to {orientation}"

        obs, reward, success = helpers["perform_action"](env, "get_output", 0)

        if success and reward > 0:
            collection_reward = reward
            break

    assert collection_reward > 0, f"Heart collection should give positive reward, got {collection_reward}"
    print(f"✅ Heart collection gave reward of {collection_reward}")


def test_multiple_heart_collections(heart_env, heart_game_map, heart_helpers):
    """Test collecting multiple hearts from altar."""
    env, obs = heart_env(heart_game_map)
    helpers = heart_helpers

    # Setup
    helpers["wait_for_heart_production"](env, steps=5)
    move_result = move(env, Orientation.RIGHT)
    assert move_result["success"], "Should be able to move to altar"

    # Find working orientation
    working_orientation = None
    for orientation in [Orientation.UP, Orientation.DOWN, Orientation.LEFT, Orientation.RIGHT]:
        rotate_result = rotate(env, orientation)
        assert rotate_result["success"], f"Should be able to rotate to {orientation}"

        obs, reward, success = helpers["perform_action"](env, "get_output", 0)

        if success and reward > 0:
            working_orientation = orientation
            break

    assert working_orientation is not None, "Should find an orientation that allows heart collection"

    # Try multiple collections
    total_hearts_collected = helpers["get_agent_hearts"](env, obs[0])
    collections_attempted = 0
    max_attempts = 5

    for attempt in range(max_attempts):
        # Wait for more heart production
        helpers["wait_for_heart_production"](env, steps=3)

        # Ensure correct orientation
        rotate_result = rotate(env, working_orientation)
        assert rotate_result["success"], "Should maintain correct orientation"

        # Try collection
        obs, reward, success = helpers["perform_action"](env, "get_output", 0)
        collections_attempted += 1

        current_hearts = helpers["get_agent_hearts"](env, obs[0])

        if current_hearts > total_hearts_collected:
            total_hearts_collected = current_hearts
            print(f"Collection {attempt + 1}: Successfully collected heart (total: {total_hearts_collected})")
        else:
            print(f"Collection {attempt + 1}: No hearts collected this attempt")

    print(f"Final result: {total_hearts_collected} hearts collected in {collections_attempted} attempts")

    # Should collect at least one heart
    assert total_hearts_collected > 0, f"Should collect at least one heart, got {total_hearts_collected}"
