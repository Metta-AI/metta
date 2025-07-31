#!/usr/bin/env python3

import copy

import numpy as np

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


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


def test_visitation_counts_configurable():
    """Test that visitation counts can be enabled/disabled via configuration."""

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

    def count_visitation_features(obs):
        """Count visitation count feature tokens in observations."""
        center_location = 34  # 5x5 obs, center is (2,2) = 34 when packed

        total_count = 0
        num_agents = obs.shape[0]
        for agent_idx in range(num_agents):
            for token_idx in range(obs.shape[1]):
                if obs[agent_idx, token_idx, 0] == center_location:
                    feature_id = obs[agent_idx, token_idx, 1]
                    if feature_id == 14:  # VisitationCounts
                        total_count += 1
        return total_count

    # Test configuration with visitation counts enabled
    game_config_with_visitation = {
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
            "visitation_counts": True,  # Explicitly enabled
        },
    }

    # Test configuration with visitation counts disabled (use deep copy)
    game_config_without_visitation = copy.deepcopy(game_config_with_visitation)
    game_config_without_visitation["global_obs"]["visitation_counts"] = False

    # Test with visitation counts enabled
    env_with = MettaGrid(from_mettagrid_config(game_config_with_visitation), map_data, 42)
    obs, _ = env_with.reset()

    visitation_count_with = count_visitation_features(obs)
    print(f"With visitation counts enabled: found {visitation_count_with} features")
    assert visitation_count_with == 5, f"Expected 5 visitation count features when enabled, got {visitation_count_with}"

    # Test with visitation counts disabled
    env_without = MettaGrid(from_mettagrid_config(game_config_without_visitation), map_data, 42)
    obs, _ = env_without.reset()

    visitation_count_without = count_visitation_features(obs)
    print(f"With visitation counts disabled: found {visitation_count_without} features")
    assert visitation_count_without == 0, (
        f"Expected 0 visitation count features when disabled, got {visitation_count_without}"
    )

    print("Visitation counts configuration test completed successfully!")


def test_visitation_counts_default_behavior():
    """Test that visitation counts are enabled by default when not specified."""

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

    # Test configuration without specifying visitation_counts (should default to True)
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
            # visitation_counts not specified - should default to True
        },
    }

    env_default = MettaGrid(from_mettagrid_config(game_config_default), map_data, 42)
    obs, _ = env_default.reset()

    # Count visitation count features (feature ID 14)
    visitation_features_default = []
    for i in range(obs.shape[1]):
        if obs[0, i, 1] == 14:  # feature_id == 14 (VisitationCounts)
            visitation_features_default.append(obs[0, i, 2])  # value

    print(f"With default visitation counts: found {len(visitation_features_default)} features")
    assert len(visitation_features_default) == 5, (
        f"Expected 5 visitation count features by default, got {len(visitation_features_default)}"
    )

    print("Visitation counts default behavior test completed successfully!")


if __name__ == "__main__":
    test_visitation_counts()
    test_visitation_counts_configurable()
    test_visitation_counts_default_behavior()
    print("All tests passed!")
