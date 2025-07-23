#!/usr/bin/env python3
"""Test script for reward-based termination functionality."""

import numpy as np
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import MettaGridEnv


def test_reward_termination():
    """Test that episodes terminate when total reward reaches threshold."""

    # Load the test configuration
    config = OmegaConf.load("configs/env/mettagrid/tests/test_reward_termination.yaml")

    # Create a simple curriculum
    curriculum = SingleTaskCurriculum("test", config.game)

    # Create the environment
    env = MettaGridEnv(curriculum=curriculum, render_mode=None)

    # Reset the environment
    obs, info = env.reset()

    print(f"Episode started. Termination threshold: {config.game.termination_reward_threshold}")

    step_count = 0
    total_reward = 0.0

    # Run the episode
    while not env.done:
        # Take a random action
        action = np.array([[0, 0]])  # noop action
        obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1
        total_reward += reward.sum()

        print(f"Step {step_count}: Reward = {reward.sum():.2f}, Total = {total_reward:.2f}")

        if terminated.any():
            print("Episode terminated due to reward threshold!")
            break
        elif truncated.any():
            print("Episode truncated due to max steps!")
            break

    print(f"Episode ended after {step_count} steps with total reward {total_reward:.2f}")

    # Verify that termination worked as expected
    termination_threshold = config.game.termination_reward_threshold
    if termination_threshold is not None:
        expected_termination = total_reward >= termination_threshold
        print(f"Expected termination: {expected_termination}")
        print(f"Actual termination: {terminated.any()}")

        if expected_termination and not terminated.any():
            print("ERROR: Episode should have terminated but didn't!")
            return False
        elif not expected_termination and terminated.any():
            print("ERROR: Episode terminated unexpectedly!")
            return False
        else:
            print("SUCCESS: Termination behavior is correct!")
            return True
    else:
        print("No termination threshold set, skipping verification")
        return True


if __name__ == "__main__":
    success = test_reward_termination()
    exit(0 if success else 1)
