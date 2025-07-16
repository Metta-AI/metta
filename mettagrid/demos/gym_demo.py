#!/usr/bin/env python3
"""
Demo script for Gymnasium integration with MettaGrid.

This script demonstrates how to use the MettaGridGymEnv with Gymnasium's
standard environment interface.
"""

import os
import sys

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.gym_env import MettaGridGymEnv, SingleAgentMettaGridGymEnv

# Add the src directory to the path to ensure imports work correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, "..", "src"))
sys.path.insert(0, src_dir)


def create_simple_config():
    """Create a simple navigation configuration."""
    return DictConfig(
        {
            "game": {
                "max_steps": 100,
                "num_agents": 2,
                "obs_width": 7,
                "obs_height": 7,
                "num_observation_tokens": 50,
                "inventory_item_names": ["ore_red", "ore_blue", "battery_red", "battery_blue", "heart"],
                "groups": {"agent": {"id": 0, "sprite": 0}},
                "agent": {
                    "default_resource_limit": 10,
                    "resource_limits": {"heart": 255},
                    "freeze_duration": 5,
                    "rewards": {"heart": 1.0},
                },
                "actions": {
                    "noop": {"enabled": True},
                    "move": {"enabled": True},
                    "rotate": {"enabled": True},
                    "put_items": {"enabled": True},
                    "get_items": {"enabled": True},
                    "attack": {"enabled": True},
                    "swap": {"enabled": True},
                    "change_color": {"enabled": False},
                    "change_glyph": {"enabled": False, "number_of_glyphs": 0},
                },
                "objects": {
                    "wall": {"type_id": 1, "swappable": False},
                },
                "map_builder": {
                    "_target_": "metta.mettagrid.room.random.Random",
                    "agents": 2,
                    "width": 16,
                    "height": 16,
                    "border_width": 1,
                    "objects": {},
                },
            }
        }
    )


def run_multi_agent_demo():
    """Run multi-agent Gymnasium demo."""
    print("Multi-Agent Gymnasium Demo")
    print("-" * 30)

    # Create config and curriculum
    config = create_simple_config()
    curriculum = SingleTaskCurriculum("gym_multi_demo", config)

    # Create environment
    print("Creating multi-agent Gymnasium environment...")
    env = MettaGridGymEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
        single_agent=False,
    )

    print(f"Number of agents: {env.num_agents}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Max steps: {env.max_steps}")

    # Run episode
    print("\nRunning episode...")
    obs, info = env.reset(seed=42)
    print(f"Initial obs shape: {obs.shape}")

    total_reward = 0
    step = 0

    while not env.done and step < 50:
        # Random actions
        actions = np.random.randint(0, 2, size=(env.num_agents, 2), dtype=np.int32)

        obs, rewards, terminals, truncations, info = env.step(actions)

        total_reward += rewards.sum()
        step += 1

        if step % 10 == 0:
            print(f"Step {step}: Total reward = {total_reward:.2f}")

    print("\nEpisode completed!")
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Done: {env.done}")

    env.close()


def run_single_agent_demo():
    """Run single-agent Gymnasium demo."""
    print("\nSingle-Agent Gymnasium Demo")
    print("-" * 30)

    # Create config and curriculum for single agent
    config = create_simple_config()
    config.game.num_agents = 1
    config.game.map_builder.agents = 1
    curriculum = SingleTaskCurriculum("gym_single_demo", config)

    # Create environment
    print("Creating single-agent Gymnasium environment...")
    env = SingleAgentMettaGridGymEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    print(f"Number of agents: {env.num_agents}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Max steps: {env.max_steps}")

    # Run episode
    print("\nRunning episode...")
    obs, info = env.reset(seed=42)
    print(f"Initial obs shape: {obs.shape}")

    total_reward = 0.0
    step = 0

    while not env.done and step < 50:
        # Random action
        action = np.random.randint(0, 2, size=2, dtype=np.int32)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step += 1

        if step % 10 == 0:
            print(f"Step {step}: Total reward = {total_reward:.2f}")

    print("\nEpisode completed!")
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Done: {env.done}")

    env.close()


def main():
    """Run Gymnasium demos."""
    print("Gymnasium MettaGrid Demo")
    print("=" * 40)

    run_multi_agent_demo()
    run_single_agent_demo()

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
