#!/usr/bin/env python3

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


if __name__ == "__main__":
    test_visitation_counts()
