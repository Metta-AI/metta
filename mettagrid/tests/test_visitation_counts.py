#!/usr/bin/env python3

import copy
import time

import numpy as np
import pytest

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


def extract_visitation_features(obs):
    """Extract visitation count feature tokens from observations."""
    features = []
    num_agents = obs.shape[0]
    for agent_idx in range(num_agents):
        for token_idx in range(obs.shape[1]):
            feature_id = obs[agent_idx, token_idx, 1]
            if feature_id == 14:  # VisitationCounts feature ID
                features.append(obs[agent_idx, token_idx])
    return features


def count_visitation_features_at_center(obs):
    """Count visitation count feature tokens at center location in observations."""
    center_location = 34  # 5x5 obs, center is (2,2) = 34 when packed (assuming 5x5 default)

    total_count = 0
    num_agents = obs.shape[0]
    for agent_idx in range(num_agents):
        for token_idx in range(obs.shape[1]):
            if obs[agent_idx, token_idx, 0] == center_location:
                feature_id = obs[agent_idx, token_idx, 1]
                if feature_id == 14:  # VisitationCounts
                    total_count += 1
    return total_count


def get_agent_position(env):
    """Get the position of the first agent in the environment."""
    # This is a simple implementation that finds the agent position from the grid
    # In a real implementation, this might be available directly from the environment
    # For testing purposes, we'll assume agent starts at (3,3) in our 7x7 test grid
    return (3, 3)


def create_test_env(visitation_counts_enabled=True):
    """Create test environment with or without visitation counts."""
    game_config = {
        "max_steps": 1000,
        "num_agents": 1,
        "episode_truncates": True,
        "obs_width": 11,
        "obs_height": 11,
        "inventory_item_names": ["wood", "stone"],
        "num_observation_tokens": 200,
        "actions": {
            "move": {"enabled": True},
        },
        "objects": {"wall": {"type_id": 1}},
        "agent": {},
        "groups": {"test_group": {"id": 0, "props": {}}},
        "global_obs": {
            "episode_completion_pct": True,
            "last_action": True,
            "last_reward": True,
            "resource_rewards": False,
            "visitation_counts": visitation_counts_enabled,
        },
    }

    # Create a simple map with agent in the middle
    map_data = [
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "agent.test_group", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
    ]

    return MettaGrid(from_mettagrid_config(game_config), map_data, 42)


def test_visitation_performance():
    """Test performance difference between enabled and disabled visitation counts."""

    print("Testing performance with visitation counts enabled...")
    env_enabled = create_test_env(visitation_counts_enabled=True)
    obs, _ = env_enabled.reset()

    # Run some steps to warm up
    actions = np.array([[0, 0]], dtype=np.int32)  # move forward
    for _ in range(10):
        obs, rewards, terminals, truncations, info = env_enabled.step(actions)

    # Time the performance
    start_time = time.time()
    for _ in range(100):
        obs, rewards, terminals, truncations, info = env_enabled.step(actions)
    enabled_time = time.time() - start_time

    print(f"With visitation counts enabled: {enabled_time:.4f} seconds for 100 steps")

    print("\nTesting performance with visitation counts disabled...")
    env_disabled = create_test_env(visitation_counts_enabled=False)
    obs, _ = env_disabled.reset()

    # Run some steps to warm up
    for _ in range(10):
        obs, rewards, terminals, truncations, info = env_disabled.step(actions)

    # Time the performance
    start_time = time.time()
    for _ in range(100):
        obs, rewards, terminals, truncations, info = env_disabled.step(actions)
    disabled_time = time.time() - start_time

    print(f"With visitation counts disabled: {disabled_time:.4f} seconds for 100 steps")

    # Calculate performance improvement
    if enabled_time > 0:
        improvement = ((enabled_time - disabled_time) / enabled_time) * 100
        print(f"\nPerformance improvement: {improvement:.2f}% faster when disabled")

    print("\nPerformance test completed!")


def test_visitation_counts():
    """Test that visitation counts are working correctly."""

    # Create a simple game config using dictionary format
    game_config = {
        "max_steps": 100,
        "num_agents": 1,
        "episode_truncates": True,
        "obs_width": 5,
        "obs_height": 5,
        "inventory_item_names": ["wood", "stone"],
        "num_observation_tokens": 100,
        "actions": {
            "move": {"enabled": True},
        },
        "objects": {"wall": {"type_id": 1}},
        "agent": {},
        "groups": {"test_group": {"id": 0, "props": {}}},
        "global_obs": {
            "episode_completion_pct": True,
            "last_action": True,
            "last_reward": True,
            "resource_rewards": False,
            "visitation_counts": True,
        },
    }

    # Create a simple map with just an agent in the middle
    map_data = [
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "agent.test_group", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
    ]

    # Create the environment
    env = MettaGrid(from_mettagrid_config(game_config), map_data, 42)

    # Reset the environment
    obs, info = env.reset()

    print("Initial observation shape:", obs.shape)
    print("Number of agents:", env.num_agents)

    # Get the agent's observation features
    agent_obs = obs[0]  # First agent's observation

    # Find visitation count features (feature ID 14)
    visitation_features = []
    for i in range(agent_obs.shape[0]):
        if agent_obs[i][1] == 14:  # feature_id == 14 (VisitationCounts)
            visitation_features.append(agent_obs[i][2])  # value

    print(f"Initial visitation counts: {visitation_features}")
    print("Expected: [0, 0, 0, 0, 0] (center, up, down, left, right)")

    # Assert that we found exactly 5 visitation count features
    assert len(visitation_features) == 5, f"Expected 5 visitation count features, got {len(visitation_features)}"
    assert visitation_features == [0, 0, 0, 0, 0], (
        f"Expected initial visitation counts [0,0,0,0,0], got {visitation_features}"
    )

    # Debug: check initial agent position and visitation grid
    grid_objects = env.grid_objects()
    for _obj_id, obj in grid_objects.items():
        if "agent_id" in obj:
            print(f"Initial agent position: ({obj['r']}, {obj['c']})")
            break

    # Take a few move actions
    actions = np.array([[0, 0]], dtype=np.int32)  # move forward

    for step in range(3):
        obs, rewards, terminals, truncations, info = env.step(actions)
        agent_obs = obs[0]

        # Find visitation count features
        visitation_features = []
        for i in range(agent_obs.shape[0]):
            if agent_obs[i][1] == 14:  # feature_id == 14 (VisitationCounts)
                visitation_features.append(agent_obs[i][2])  # value

        print(f"Step {step + 1} visitation counts: {visitation_features}")

        # Debug: print all features to see what's available
        if step == 0:
            print("Debug: All features in first step:")
            for i in range(min(20, agent_obs.shape[0])):  # Print first 20 tokens
                if agent_obs[i][0] != 0xFF:  # Not empty
                    print(
                        f"  Token {i}: location={agent_obs[i][0]}, "
                        f"feature_id={agent_obs[i][1]}, value={agent_obs[i][2]}"
                    )

            # Search for visitation count features across all tokens
            print("Searching for visitation count features (feature_id=14) across all tokens:")
            for i in range(agent_obs.shape[0]):
                if agent_obs[i][0] != 0xFF and agent_obs[i][1] == 14:  # Not empty and feature_id=14
                    print(f"  Found visitation count at token {i}: location={agent_obs[i][0]}, value={agent_obs[i][2]}")

            # Debug: check all feature IDs being used
            print("All feature IDs in observations:")
            feature_ids = set()
            for i in range(agent_obs.shape[0]):
                if agent_obs[i][0] != 0xFF:  # Not empty
                    feature_ids.add(agent_obs[i][1])
            print(f"  Feature IDs found: {sorted(feature_ids)}")

            # Check if move action was successful
            action_success = env.action_success()
            print(f"Move action success: {action_success}")

            # Check agent position
            grid_objects = env.grid_objects()
            for _obj_id, obj in grid_objects.items():
                if "agent_id" in obj:
                    print(f"Agent position: ({obj['r']}, {obj['c']})")
                    break

    print("Test completed successfully!")
    print("Visitation counts should show the agent's movement history.")


def test_visitation_counts_default_behavior():
    """Test that visitation counts are disabled by default when not specified."""

    # Create a simple map with just an agent in the middle
    map_data = [
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "agent.test_group", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
    ]

    # Test configuration without specifying visitation_counts (should default to False)
    game_config_default = {
        "max_steps": 100,
        "num_agents": 1,
        "episode_truncates": True,
        "obs_width": 5,
        "obs_height": 5,
        "inventory_item_names": ["wood", "stone"],
        "num_observation_tokens": 100,
        "actions": {
            "move": {"enabled": True},
        },
        "objects": {"wall": {"type_id": 1}},
        "agent": {},
        "groups": {"test_group": {"id": 0, "props": {}}},
        "global_obs": {
            "episode_completion_pct": True,
            "last_action": True,
            "last_reward": True,
            "resource_rewards": False,
            # visitation_counts not specified - should default to False
        },
    }

    env_default = MettaGrid(from_mettagrid_config(game_config_default), map_data, 42)
    obs, _ = env_default.reset()

    features = extract_visitation_features(obs)
    assert len(features) == 0, "Visitation counts should be disabled by default"


def test_visitation_counts_configurable(base_config, simple_map):
    """Test configuration variations for visitation counts."""
    # Test enabled
    config_enabled = copy.deepcopy(base_config)
    config_enabled["global_obs"]["visitation_counts"] = True
    env = MettaGrid(from_mettagrid_config(config_enabled), simple_map, 42)
    obs, _ = env.reset()
    count = count_visitation_features_at_center(obs)
    assert count == 5, f"Expected 5 features when enabled, got {count}"

    # Test disabled
    config_disabled = copy.deepcopy(base_config)
    config_disabled["global_obs"]["visitation_counts"] = False
    env = MettaGrid(from_mettagrid_config(config_disabled), simple_map, 42)
    obs, _ = env.reset()
    count = count_visitation_features_at_center(obs)
    assert count == 0, f"Expected 0 features when disabled, got {count}"

    # Test default (not specified)
    env = MettaGrid(from_mettagrid_config(base_config), simple_map, 42)
    obs, _ = env.reset()
    count = count_visitation_features_at_center(obs)
    assert count == 0, f"Expected 0 features by default, got {count}"


@pytest.fixture
def base_config():
    """Base configuration for visitation tests."""
    return {
        "max_steps": 100,
        "num_agents": 1,
        "obs_width": 5,
        "obs_height": 5,
        "num_observation_tokens": 100,
        "inventory_item_names": ["wood", "stone"],
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
        },
        "groups": {"test_group": {"id": 0, "props": {}}},
        "objects": {"wall": {"type_id": 1}},
        "agent": {"rewards": {}},
        "global_obs": {
            "episode_completion_pct": True,
            "last_action": True,
            "last_reward": True,
            "resource_rewards": False,
        },
    }


@pytest.fixture
def simple_map():
    """Simple map for testing."""
    return [
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "agent.test_group", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
    ]


@pytest.fixture
def env_with_visitation(base_config, simple_map):
    """Environment with visitation counts enabled."""
    config = copy.deepcopy(base_config)
    config["global_obs"]["visitation_counts"] = True
    return MettaGrid(from_mettagrid_config(config), simple_map, 42)


@pytest.fixture
def performance_config(base_config):
    """Config for performance testing with more steps."""
    config = copy.deepcopy(base_config)
    config["max_steps"] = 1000
    config["obs_width"] = 11
    config["obs_height"] = 11
    config["num_observation_tokens"] = 200
    return config


def _median_runtime(env, move_action, warmup_steps=10, test_steps=200, reps=15):
    # warmup
    for _ in range(warmup_steps):
        env.step(move_action)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(test_steps):
            env.step(move_action)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def test_visitation_performance_impact(performance_config, simple_map):
    """Disabled visitation should not be materially slower."""
    move_action = np.array([[0, 0]], dtype=np.int32)

    # enabled
    cfg_on = copy.deepcopy(performance_config)
    cfg_on["global_obs"]["visitation_counts"] = True
    env_enabled = MettaGrid(from_mettagrid_config(cfg_on), simple_map, 42)
    env_enabled.reset()
    enabled_time = _median_runtime(env_enabled, move_action)

    # disabled
    cfg_off = copy.deepcopy(performance_config)
    cfg_off["global_obs"]["visitation_counts"] = False
    env_disabled = MettaGrid(from_mettagrid_config(cfg_off), simple_map, 42)
    env_disabled.reset()
    disabled_time = _median_runtime(env_disabled, move_action)

    # allow small jitter (≤10% slowdown)
    assert disabled_time <= enabled_time * 1.1, (
        f"Disabled visitation unexpectedly slower: "
        f"enabled={enabled_time:.6f}s, disabled={disabled_time:.6f}s, "
        f"delta={(disabled_time / enabled_time - 1) * 100:.2f}%"
    )


def test_agent_movement_tracking(env_with_visitation):
    """Test that visitation counts work with agent movement."""
    obs, _ = env_with_visitation.reset()

    # Get initial observation
    initial_visitation_count = len(extract_visitation_features(obs))

    # Move and check that visitation features are present
    move_action = np.array([[0, 0]], dtype=np.int32)
    obs, _, _, _, _ = env_with_visitation.step(move_action)

    # Check that visitation features are still present after movement
    final_visitation_count = len(extract_visitation_features(obs))
    assert final_visitation_count > 0, "Visitation features should be present when enabled"
    assert final_visitation_count >= initial_visitation_count, "Visitation features should persist or increase"


def test_observation_structure(base_config, simple_map):
    """Test the structure of observations with visitation counts."""
    # Create environment without visitation counts (default)
    env = MettaGrid(from_mettagrid_config(base_config), simple_map, 42)
    obs, _ = env.reset()

    # Check observation shape
    assert len(obs.shape) == 3, f"Expected 3D observation array, got shape {obs.shape}"
    assert obs.shape[0] == 1, f"Expected 1 agent, got {obs.shape[0]}"

    # Count visitation count features (feature ID 14)
    visitation_features_default = []
    for i in range(obs.shape[1]):
        if obs[0, i, 1] == 14:  # feature_id == 14 (VisitationCounts)
            visitation_features_default.append(obs[0, i, 2])  # value

    print(f"With default visitation counts: found {len(visitation_features_default)} features")
    assert len(visitation_features_default) == 0, (
        f"Expected 0 visitation count features by default (disabled), got {len(visitation_features_default)}"
    )

    print("Visitation counts default behavior test completed successfully!")


if __name__ == "__main__":
    test_visitation_counts()
    test_visitation_counts_configurable()
    test_visitation_counts_default_behavior()
    test_visitation_performance()
    print("All tests passed!")
