#!/usr/bin/env python3
"""
Demo script for PettingZoo integration with MettaGrid.

This script demonstrates how to use the MettaGridPettingZooEnv with PettingZoo's
ParallelEnv interface.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.pettingzoo_env import MettaGridPettingZooEnv


def create_simple_config():
    """Create a simple navigation configuration."""
    return DictConfig(
        {
            "game": {
                "max_steps": 100,
                "num_agents": 3,
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
                    "agents": 3,
                    "width": 16,
                    "height": 16,
                    "border_width": 1,
                    "objects": {},
                },
            }
        }
    )


def main():
    """Run PettingZoo demo."""
    print("PettingZoo MettaGrid Demo")
    print("=" * 40)

    # Create config and curriculum
    config = create_simple_config()
    curriculum = SingleTaskCurriculum("pettingzoo_demo", config)

    # Create environment
    print("Creating PettingZoo environment...")
    env = MettaGridPettingZooEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    print(f"Possible agents: {env.possible_agents}")
    print(f"Max agents: {env.max_num_agents}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Max steps: {env.max_steps}")

    # Run episode
    print("\nRunning episode...")
    observations, infos = env.reset(seed=42)
    print(f"Active agents: {env.agents}")
    print(f"Initial obs shapes: {[(agent, obs.shape) for agent, obs in observations.items()]}")

    total_rewards = {agent: 0.0 for agent in env.possible_agents}
    step = 0

    while env.agents and step < 50:
        # Random actions for active agents
        actions = {}
        for agent in env.agents:
            actions[agent] = np.random.randint(0, 2, size=2, dtype=np.int32)

        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Update total rewards
        for agent, reward in rewards.items():
            total_rewards[agent] += reward

        step += 1

        if step % 10 == 0:
            active_rewards = {agent: total_rewards[agent] for agent in env.agents}
            print(f"Step {step}: Active agents: {len(env.agents)}, Rewards: {active_rewards}")

    print("\nEpisode completed!")
    print(f"Total steps: {step}")
    print(f"Final rewards: {total_rewards}")
    print(f"Remaining agents: {env.agents}")

    env.close()
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
