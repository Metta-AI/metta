#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "gymnasium>=0.29.0",
#     "omegaconf",
#     "typing-extensions",
#     "pydantic",
# ]
# ///

"""
Simple RL Demo for MettaGrid Environment

This demo shows how to:
1. Import and create a MettaGrid environment
2. Get observations and understand the observation space
3. Send random actions and understand the action space
4. Run a basic RL loop

This is a minimal example to get started with the MettaGrid environment.
For more advanced usage, see the demos in mettagrid/demos/
"""

from metta.mettagrid import MettaGridGymEnv
from metta.mettagrid.config.envs import make_navigation


def main():
    """Run a simple RL demo with random actions."""
    print("=== MettaGrid Simple RL Demo ===")
    print()

    env = MettaGridGymEnv(
        mg_config=make_navigation(num_agents=1),
        render_mode=None,
    )

    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Max episode steps: {env.max_steps}")
    print()

    observation, info = env.reset(seed=42)
    print(f"Initial observation shape: {observation.shape}")
    print(f"Initial observation range: [{observation.min():.3f}, {observation.max():.3f}]")
    print()

    total_reward = 0
    step_count = 0
    max_steps = 100

    print("Starting RL loop with random actions...")
    print("=" * 50)

    for step in range(max_steps):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        if step % 10 == 0:
            print(
                f"Step {step:3d}: action={action}, reward={reward:.3f}, "
                f"obs_shape={observation.shape}, terminated={terminated}"
            )

        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"Reason: {'terminated' if terminated else 'truncated'}")

            observation, info = env.reset()
            print("Environment reset for new episode")
            print()

    print("=" * 50)
    print("Demo completed!")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Average reward per step: {total_reward / step_count:.3f}")

    env.close()
    print("\nEnvironment closed successfully.")


if __name__ == "__main__":
    main()
