#!/usr/bin/env python3
"""Quick test to verify change_glyph action works correctly."""

import numpy as np

from mettagrid.config.mettagrid_c_config import from_mettagrid_config
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    ChangeGlyphActionConfig,
    GameConfig,
    WallConfig,
)
from mettagrid.map_builder.utils import create_grid
from mettagrid.mettagrid_c import MettaGrid

# Create a minimal environment with change_glyph enabled
game_config = GameConfig(
    max_steps=100,
    num_agents=2,
    obs_width=5,
    obs_height=5,
    num_observation_tokens=10,
    resource_names=["energy"],
    actions=ActionsConfig(
        noop=ActionConfig(enabled=True),
        move=ActionConfig(enabled=True),
        change_glyph=ChangeGlyphActionConfig(enabled=True, number_of_glyphs=10),
    ),
    objects={"wall": WallConfig(type_id=1)},
    agent=AgentConfig(),
)

# Create simple map using string format
game_map = create_grid(10, 10)
game_map[0, :] = "wall"
game_map[-1, :] = "wall"
game_map[:, 0] = "wall"
game_map[:, -1] = "wall"
game_map[4, 4] = "agent.red"
game_map[5, 5] = "agent.red"

# Create environment
env = MettaGrid(from_mettagrid_config(game_config), game_map.tolist(), 42)
env.reset()

# Get action indices
action_names = env.action_names
change_glyph_action = f"change_glyph_{5}"
if change_glyph_action not in action_names:
    raise RuntimeError(f"{change_glyph_action} not found in action set: {action_names}")
change_glyph_idx = action_names.index(change_glyph_action)
noop_idx = action_names.index("noop")

print(f"Action names: {env.action_names}")
print(f"change_glyph index: {change_glyph_idx}")

# Get initial glyph values
grid_objects = env.grid_objects()
agent_0 = None
agent_1 = None
for obj in grid_objects.values():
    if obj.get("agent_id") == 0:
        agent_0 = obj
    elif obj.get("agent_id") == 1:
        agent_1 = obj

print("\nInitial state:")
print(f"Agent 0 glyph: {agent_0.get('glyph')}")
print(f"Agent 1 glyph: {agent_1.get('glyph')}")

# Agent 0 changes glyph to 5, Agent 1 does nothing
actions = np.array([change_glyph_idx, noop_idx], dtype=np.int32)
print(f"\nSending actions: {actions}")

obs, rewards, terminals, truncations, info = env.step(actions)

# Get updated glyph values
grid_objects = env.grid_objects()
agent_0 = None
agent_1 = None
for obj in grid_objects.values():
    if obj.get("agent_id") == 0:
        agent_0 = obj
    elif obj.get("agent_id") == 1:
        agent_1 = obj

print("\nAfter change_glyph action:")
print(f"Agent 0 glyph: {agent_0.get('glyph')} (expected: 5)")
print(f"Agent 1 glyph: {agent_1.get('glyph')} (expected: unchanged)")

# Verify
if agent_0.get("glyph") == 5:
    print("\n✓ change_glyph action works correctly!")
else:
    print(f"\n✗ change_glyph action failed! Agent 0 glyph is {agent_0.get('glyph')}, expected 5")
