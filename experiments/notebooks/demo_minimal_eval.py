#!/usr/bin/env -S uv run
"""Minimal evaluation of the built-in opportunistic policy.

This mirrors the simple evaluation loop used in the Hello-World notebook:
1. Build the hallway environment (same config as demo_minimal_approach.py)
2. Run N episodes for a fixed number of steps each
3. After each episode tally `ore_red` in the agent's inventory – that is the *score*
4. Report per-episode scores and a summary

Why not use the full Simulation / sim.py pipeline?
The opportunistic policy is a lightweight helper that doesn’t implement the
full PolicyAgent interface expected by the simulation infrastructure.
For clarity we keep this evaluation self-contained.
"""

from __future__ import annotations

import contextlib
import io
import statistics
from typing import List

from metta.interface.environment import _get_default_env_config
from omegaconf import OmegaConf
from tools.renderer import get_policy, setup_environment

# ---------------------------------------------------------------------------
# 1. Build hallway environment config (same as demo_minimal_approach.py)
# ---------------------------------------------------------------------------
hallway_map = """###########
#@.......m#
###########"""

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

# Mine tweaks to make ore plentiful
mine = env_dict["game"]["objects"]["mine_red"]
mine["initial_resource_count"] = 1
mine["conversion_ticks"] = 4
mine["cooldown"] = 0
mine["max_output"] = 2

# Reward for ore in inventory
env_dict["game"]["agent"]["rewards"]["inventory"]["ore_red"] = 1.0

cfg = OmegaConf.create(
    {
        "env": env_dict,
        "renderer_job": {
            "policy_type": "opportunistic",
            "num_steps": 300,  # ticks per episode (same as observation cell)
            "num_agents": 1,
            "sleep_time": 0.0,  # no delay for fast eval
        },
    }
)

# ---------------------------------------------------------------------------
# 2. Set up environment & policy (suppress ANSI prints from renderer)
# ---------------------------------------------------------------------------
with contextlib.redirect_stdout(io.StringIO()):
    env, _ = setup_environment(cfg)
    policy = get_policy(cfg.renderer_job.policy_type, env, cfg)

# ---------------------------------------------------------------------------
# 3. Run evaluation loop
# ---------------------------------------------------------------------------
N_EPISODES = 30  # keep runtime quick
scores: List[int] = []

for ep in range(1, N_EPISODES + 1):
    obs, _ = env.reset()
    # Run fixed number of steps
    for _ in range(cfg.renderer_job.num_steps):
        actions = policy.predict(obs)
        obs, _, _, _, _ = env.step(actions)

    # Count ore in inventory after episode
    agent_obj = next(o for o in env.grid_objects.values() if o.get("agent_id") == 0)
    inv = {
        env.inventory_item_names[idx]: count
        for idx, count in agent_obj.get("inventory", {}).items()
    }
    ore = int(inv.get("ore_red", 0))
    scores.append(ore)
    print(f"Episode {ep:2d}/{N_EPISODES}: ore_red = {ore}")

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
mean = statistics.mean(scores)
std = statistics.pstdev(scores) if len(scores) > 1 else 0
print("\n=== Summary ===")
print(f"Mean ore_red: {mean:.2f} ± {std:.2f} over {N_EPISODES} episodes")
print(f"Min / Max : {min(scores)} / {max(scores)}")

env.close()
