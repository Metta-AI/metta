#!/usr/bin/env -S uv run

import time  # for per-step delay

from metta.interface.environment import _get_default_env_config
from omegaconf import OmegaConf
from tools.renderer import get_policy, setup_environment

# Define a simple hallway map with more ore
hallway_map = """###########
#@.......m#
###########"""
# Build environment dict from default and apply overrides
env_dict = _get_default_env_config(num_agents=1, width=11, height=3)
# Inline ASCII map builder
env_dict["game"]["map_builder"] = {
    "_target_": "metta.map.mapgen.MapGen",
    "border_width": 0,
    "root": {
        "type": "metta.map.scenes.inline_ascii.InlineAscii",
        "params": {"data": hallway_map},
    },
}
# Override mine settings
env_dict["game"]["objects"]["mine_red"]["initial_resource_count"] = 1
env_dict["game"]["objects"]["mine_red"]["conversion_ticks"] = 4
env_dict["game"]["objects"]["mine_red"]["cooldown"] = 0
env_dict["game"]["objects"]["mine_red"]["output_limit"] = 2
# Override reward settings
env_dict["game"]["agent"]["rewards"]["inventory"]["ore_red"] = 1.0
# Single OmegaConf config
cfg = OmegaConf.create(
    {
        "env": env_dict,
        "renderer_job": {
            "policy_type": "opportunistic",
            "num_steps": 300,
            "num_agents": 1,
            "sleep_time": 0.05,
        },
    }
)
# Setup environment and policy
env, _ = setup_environment(cfg)
policy = get_policy(cfg.renderer_job.policy_type, env, cfg)

# Run simulation
obs, info = env.reset()
total_reward = 0.0
for step in range(cfg.renderer_job.num_steps):
    actions = policy.predict(obs)
    obs, rewards, terminals, truncations, info = env.step(actions)

    # Extract inventory for agent 0
    agent_obj = next(o for o in env.grid_objects.values() if o.get("agent_id") == 0)
    inv = {
        env.inventory_item_names[idx]: count
        for idx, count in agent_obj.get("inventory", {}).items()
    }
    total_reward = inv.get("ore_red", 0)

    # Clear screen and display
    print("\033[2J\033[H", end="")
    print(
        f"Score: {total_reward:.1f} | Inventory: {total_reward} ore(s) | Step: {step + 1}/{cfg.renderer_job.num_steps}"
    )
    # Print map buffer without additional clearing
    renderer = env._renderer
    if renderer is not None:
        buffer = renderer.get_buffer(env.grid_objects)
        print(buffer)

    time.sleep(cfg.renderer_job.sleep_time)

env.close()
print(f"\nFinal total reward: {total_reward}")
