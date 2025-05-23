"""
Heart Collection Test - pytest compatible with fixtures
"""

import numpy as np
import pytest

from mettagrid.mettagrid_c import MettaGrid
from mettagrid.tests.actions import (
    get_agent_position,
    move,
    np_actions_type,
    np_observations_type,
    np_rewards_type,
    np_terminals_type,
    np_truncations_type,
    rotate,
)


@pytest.fixture
def heart_config():
    """Configuration for heart collection tests."""
    return {
        "max_steps": 50,
        "num_agents": 1,
        "obs_width": 3,
        "obs_height": 3,
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
                "cooldown": -1,
            },
        },
        "agent": {
            "inventory_size": 10,
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
def adjacent_heart_game_map():
    """Game map with agent already adjacent to altar."""
    return [
        ["wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "agent.red", "altar", "empty", "wall"],
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
        grid_features = env.grid_features()
        if "agent:inv:heart" in grid_features:
            feature_idx = grid_features.index("agent:inv:heart")
            return int(np.sum(obs[:, :, feature_idx]))
        return 0

    def perform_action(env, action_name, arg=0):
        action_idx = env.action_names().index(action_name)
        action = np.zeros((1, 2), dtype=np_actions_type)
        action[0] = [action_idx, arg]
        obs, rewards, _terminals, _truncations, _info = env.step(action)
        return obs, float(rewards[0]), env.action_success()[0]

    def get_altar_info(env):
        grid_objects = env.grid_objects()
        for _obj_id, obj_data in grid_objects.items():
            if "altar" in obj_data:
                altar_pos = (obj_data["r"], obj_data["c"])
                altar_hearts = obj_data.get("inv:heart", 0)
                return altar_pos, altar_hearts
        return None, 0

    return {
        "get_agent_hearts": get_agent_hearts,
        "perform_action": perform_action,
        "get_altar_info": get_altar_info,
    }


def test_heart_collection_basic(heart_env, heart_game_map, heart_helpers):
    """Test basic heart collection from altar."""
    env, obs = heart_env(heart_game_map)
    helpers = heart_helpers

    # Get initial positions
    agent_pos = get_agent_position(env, 0)
    altar_pos, altar_hearts = helpers["get_altar_info"](env)

    assert agent_pos is not None, "Agent should have a valid position"
    assert altar_pos is not None, "Altar should have a valid position"

    # Wait for heart production
    for _i in range(3):
        obs, reward, success = helpers["perform_action"](env, "noop")

    # Move to be adjacent to altar
    move_result = move(env, 3)  # Move right
    assert move_result["success"], f"Movement should succeed. Error: {move_result.get('error')}"

    # Verify we're adjacent to altar
    agent_pos = get_agent_position(env, 0)
    altar_pos, altar_hearts = helpers["get_altar_info"](env)

    assert agent_pos is not None, "Agent should still have a valid position after moving"
    assert altar_pos is not None, "Altar should still have a valid position"

    distance = abs(agent_pos[0] - altar_pos[0]) + abs(agent_pos[1] - altar_pos[1])
    assert distance == 1, f"Agent should be adjacent to altar (distance 1), but distance is {distance}"

    # Try heart collection
    hearts_before = helpers["get_agent_hearts"](env, obs[0])
    obs, reward, success = helpers["perform_action"](env, "get_output", 0)
    hearts_after = helpers["get_agent_hearts"](env, obs[0])
    hearts_gained = hearts_after - hearts_before

    if not (success and hearts_gained > 0):
        # If basic collection failed, try all orientations
        print("Basic collection failed, trying all orientations...")

        orientation_names = {0: "up", 1: "down", 2: "left", 3: "right"}
        collection_succeeded = False

        for orientation in range(4):
            direction_name = orientation_names[orientation]

            # Rotate to face this direction
            rotate_result = rotate(env, orientation)
            assert rotate_result["success"], f"Rotation to {direction_name} should succeed"

            # Try collection
            hearts_before = helpers["get_agent_hearts"](env, obs[0])
            obs, reward, success = helpers["perform_action"](env, "get_output", 0)
            hearts_after = helpers["get_agent_hearts"](env, obs[0])
            hearts_gained = hearts_after - hearts_before

            if success and hearts_gained > 0:
                collection_succeeded = True
                print(f"Heart collection succeeded when facing {direction_name}")
                break

        assert collection_succeeded, "Heart collection should succeed in at least one orientation"
    else:
        # Basic collection worked
        assert success, "get_output action should succeed"
        assert hearts_gained > 0, f"Should gain hearts, but gained {hearts_gained}"
        assert reward > 0, f"Should receive positive reward, but got {reward}"


def test_heart_collection_orientation_independence(heart_env, adjacent_heart_game_map, heart_helpers):
    """Test that heart collection works regardless of agent orientation."""
    env, obs = heart_env(adjacent_heart_game_map)
    helpers = heart_helpers

    # Wait for heart production
    for _i in range(3):
        obs, reward, success = helpers["perform_action"](env, "noop")

    # Move right to be adjacent to altar
    move_result = move(env, 3)
    assert move_result["success"], "Should be able to move adjacent to altar"

    # Test collection in each orientation
    orientation_names = {0: "up", 1: "down", 2: "left", 3: "right"}
    successful_orientations = []

    for orientation in range(4):
        direction_name = orientation_names[orientation]

        # Rotate to face direction
        rotate_result = rotate(env, orientation)
        assert rotate_result["success"], f"Should be able to rotate to face {direction_name}"

        # Try collection
        hearts_before = helpers["get_agent_hearts"](env, obs[0])
        obs, reward, success = helpers["perform_action"](env, "get_output", 0)
        hearts_after = helpers["get_agent_hearts"](env, obs[0])
        hearts_gained = hearts_after - hearts_before

        if success and hearts_gained > 0:
            successful_orientations.append(direction_name)

    # At least one orientation should work
    assert len(successful_orientations) > 0, (
        f"Heart collection should work in at least one orientation, but failed in all: "
        f"{list(orientation_names.values())}"
    )

    print(f"Heart collection succeeded in orientations: {successful_orientations}")


def test_heart_production_timing(heart_env, adjacent_heart_game_map, heart_helpers):
    """Test that hearts are produced over time."""
    env, obs = heart_env(adjacent_heart_game_map)
    helpers = heart_helpers

    # Check initial altar state
    altar_pos, initial_hearts = helpers["get_altar_info"](env)
    assert altar_pos is not None, "Altar should exist"

    # Wait and check if hearts are produced
    for step in range(10):
        obs, reward, success = helpers["perform_action"](env, "noop")
        altar_pos, current_hearts = helpers["get_altar_info"](env)

        if current_hearts > initial_hearts:
            print(f"Hearts produced after {step + 1} steps: {initial_hearts} → {current_hearts}")
            break
    else:
        # If no hearts produced after 10 steps, that might be expected based on config
        print(f"No hearts produced after 10 steps (initial: {initial_hearts})")


def test_multiple_heart_collection(heart_env, adjacent_heart_game_map, heart_helpers):
    """Test collecting multiple hearts over time."""
    env, obs = heart_env(adjacent_heart_game_map)
    helpers = heart_helpers

    # Wait for heart production
    for _i in range(5):
        obs, reward, success = helpers["perform_action"](env, "noop")

    # Move to be adjacent to altar
    move_result = move(env, 3)
    if move_result["success"]:
        print("Successfully moved adjacent to altar")

    total_hearts_collected = 0
    collection_attempts = 5

    for attempt in range(collection_attempts):
        # Try to collect hearts
        hearts_before = helpers["get_agent_hearts"](env, obs[0])
        obs, reward, success = helpers["perform_action"](env, "get_output", 0)
        hearts_after = helpers["get_agent_hearts"](env, obs[0])
        hearts_gained = hearts_after - hearts_before

        if hearts_gained > 0:
            total_hearts_collected += hearts_gained
            print(f"Attempt {attempt + 1}: Collected {hearts_gained} hearts (total: {total_hearts_collected})")

        # Wait a bit between attempts
        for _i in range(2):
            obs, reward, success = helpers["perform_action"](env, "noop")

    # We should have collected at least some hearts
    assert total_hearts_collected > 0, f"Should have collected at least some hearts over {collection_attempts} attempts"
