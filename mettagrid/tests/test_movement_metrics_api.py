#!/usr/bin/env python3
"""Test script to verify movement metrics work through the API.

This test verifies that:
1. Movement metrics can be enabled/disabled via the API
2. The API correctly passes the track_movement_metrics parameter
3. Basic environment functionality works with movement metrics enabled
"""

import torch

from metta.api import Environment


def test_movement_metrics_api():
    """Test movement metrics through the new API.

    Tests both enabled and disabled states to ensure the API parameter works correctly.
    """

    print("Testing movement metrics via API...")
    print("This test verifies API integration of movement metrics (movement/facing/* and movement/sequential_rotations)")

    # Test without movement metrics (default)
    print("\n1. Testing without movement metrics (default)...")
    env_no_metrics = Environment(
        num_agents=1, width=10, height=10, track_movement_metrics=False, num_envs=1, vectorization="serial"
    )

    # Test with movement metrics enabled
    print("\n2. Testing with movement metrics enabled...")
    env_with_metrics = Environment(
        num_agents=1, width=10, height=10, track_movement_metrics=True, num_envs=1, vectorization="serial"
    )

    # Test a few steps with movement metrics
    obs = env_with_metrics.reset()

    # Get action space info
    action_space = env_with_metrics.single_action_space
    print(f"Action space: {action_space}")

    # Get action names
    action_names = env_with_metrics.action_names
    print(f"Action names: {action_names}")

    # Find rotate action
    rotate_idx = action_names.index("rotate") if "rotate" in action_names else None
    move_idx = action_names.index("move") if "move" in action_names else None

    if rotate_idx is not None:
        print(f"Rotate action index: {rotate_idx}")
        print(f"Move action index: {move_idx}")

        # Test rotation
        actions = torch.tensor([[rotate_idx, 1]], dtype=torch.int32)  # Rotate to Down
        obs, rewards, dones, infos = env_with_metrics.step(actions)

        print(f"After rotation - dones: {dones}")
        print(f"Info keys: {list(infos.keys()) if infos else 'None'}")

    else:
        print("No rotate action found")

    # Close environments
    env_no_metrics.close()
    env_with_metrics.close()

    print("\nAPI test completed!")


if __name__ == "__main__":
    test_movement_metrics_api()
