#!/usr/bin/env python3
"""
Demo script for PufferLib integration with MettaGrid.

This script demonstrates how to use the MettaGridPufferEnv with PufferLib's
vectorized environment interface.
"""

import os
import sys

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.puffer_env import MettaGridPufferEnv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def create_simple_config():
    """Create a simple navigation configuration."""
    return DictConfig({
        "game": {
            "max_steps": 100,
            "num_agents": 2,
            "obs_width": 7,
            "obs_height": 7,
            "num_observation_tokens": 50,
            "inventory_item_names": [
                "ore_red", "ore_blue", "battery_red", "battery_blue", "heart"
            ],
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
    })

def main():
    """Run PufferLib demo."""
    print("PufferLib MettaGrid Demo")
    print("=" * 40)
    
    # Create config and curriculum
    config = create_simple_config()
    curriculum = SingleTaskCurriculum("puffer_demo", config)
    
    # Create environment
    print("Creating PufferLib environment...")
    env = MettaGridPufferEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )
    
    print(f"Number of agents: {env.num_agents}")
    print(f"Observation space: {env.single_observation_space}")
    print(f"Action space: {env.single_action_space}")
    print(f"Max steps: {env.max_steps}")
    
    # Run episode
    print("\nRunning episode...")
    obs, _ = env.reset(seed=42)
    print(f"Initial obs shape: {obs.shape}")
    
    total_reward = 0
    step = 0
    
    while not env.done and step < 50:
        # Random actions
        actions = np.random.randint(0, 2, size=(env.num_agents, 2), dtype=np.int32)
        
        obs, rewards, _, _, _ = env.step(actions)
        
        total_reward += rewards.sum()
        step += 1
        
        if step % 10 == 0:
            print(f"Step {step}: Total reward = {total_reward:.2f}")
    
    print("\nEpisode completed!")
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Done: {env.done}")
    
    env.close()
    print("Demo completed successfully!")

if __name__ == "__main__":
    main()