#!/usr/bin/env python
"""Test script to verify raylib renderer integration."""

from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import MettaGridEnv

# Create a simple environment config
env_config = DictConfig(
    {
        "game": {
            "num_agents": 1,
            "max_steps": 100,
            "obs_width": 5,
            "obs_height": 5,
            "num_observation_tokens": 25,
            "inventory_item_names": [],
            "groups": {"agent": {"id": 0}},
            "agent": {},
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "attack": {"enabled": False},
                "put_items": {"enabled": False},
                "get_items": {"enabled": False},
                "swap": {"enabled": False},
                "change_color": {"enabled": False},
                "change_glyph": {"enabled": False, "number_of_glyphs": 0},
            },
            "objects": {
                "wall": {"type_id": 1},
            },
            "map_builder": {
                "_target_": "metta.mettagrid.room.random.Random",
                "agents": 1,
                "width": 8,
                "height": 8,
                "border_width": 1,
                "objects": {},
            },
        }
    }
)

# Test different render modes
print("Testing render modes...")

# Test 1: Human renderer (NetHack style)
print("\n1. Testing 'human' renderer (NetHack):")
curriculum = SingleTaskCurriculum("test", env_config)
env = MettaGridEnv(curriculum, render_mode="human")
env.reset()
render_output = env.render()
print(f"   Renderer type: {type(env._renderer).__name__}")
print(f"   Output type: {type(render_output)}")
print(f"   Has output: {render_output is not None}")

# Test 2: Raylib renderer
print("\n2. Testing 'raylib' renderer:")
try:
    curriculum = SingleTaskCurriculum("test", env_config)
    env = MettaGridEnv(curriculum, render_mode="raylib")
    env.reset()
    render_output = env.render()
    print(f"   Renderer type: {type(env._renderer).__name__ if env._renderer else 'None'}")
    print(f"   Output type: {type(render_output)}")
    print(f"   Renderer initialized: {env._renderer is not None}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Miniscope renderer
print("\n3. Testing 'miniscope' renderer:")
curriculum = SingleTaskCurriculum("test", env_config)
env = MettaGridEnv(curriculum, render_mode="miniscope")
env.reset()
render_output = env.render()
print(f"   Renderer type: {type(env._renderer).__name__}")
print(f"   Output type: {type(render_output)}")
print(f"   Has output: {render_output is not None}")

print("\n✅ Renderer tests complete!")
