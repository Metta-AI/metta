from __future__ import annotations

import argparse
import os
import sys
import numpy as np

# Ensure src layout is importable when running directly
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from cogames.cogs_vs_clips.missions import make_game
from cogames.policy.scripted_agent_clipping import ScriptedAgentPolicy
from mettagrid import MettaGridEnv, dtype_actions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run clipping-aware scripted agent on clipped TF map")
    p.add_argument("--map", default="training_facility_clipped.map")
    p.add_argument("--cogs", type=int, default=1)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env_cfg = make_game(num_cogs=args.cogs, map_name=args.map)
    env = MettaGridEnv(env_cfg=env_cfg)

    policy = ScriptedAgentPolicy(env)
    agents = [policy.agent_policy(i) for i in range(env.num_agents)]

    obs, _ = env.reset(seed=args.seed)
    totals = np.zeros(env.num_agents, dtype=float)
    for _ in range(args.steps):
        actions = np.zeros(env.num_agents, dtype=dtype_actions)
        for i in range(env.num_agents):
            actions[i] = int(agents[i].step(obs[i]))
        obs, rewards, done, truncated, _ = env.step(actions)
        totals += rewards
        if all(done) or all(truncated):
            break

    print(f"Total reward: {totals.sum():.2f}")
    env.close()


if __name__ == "__main__":
    main()
