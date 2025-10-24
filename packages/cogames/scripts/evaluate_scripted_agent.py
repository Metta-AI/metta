#!/usr/bin/env python3
"""Evaluate ScriptedAgentPolicy across training facility maps.

Usage:
  uv run python -u packages/cogames/scripts/evaluate_scripted_agent.py \
      --cogs 1 --episodes 3 --steps 500 --seed 42

"""

from __future__ import annotations

import argparse
from typing import List

import numpy as np

from cogames.cogs_vs_clips.missions import make_game
from cogames.policy.scripted_agent import ScriptedAgentPolicy
from mettagrid import MettaGridEnv, dtype_actions

# Known-good TF maps (object names match mission objects)
TF_MAPS: List[str] = [
    "training_facility_open_1.map",
    "training_facility_open_2.map",
    "training_facility_open_3.map",
    "training_facility_tight_4.map",
    "training_facility_tight_5.map",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ScriptedAgentPolicy on TF maps")
    p.add_argument("--cogs", type=int, default=2, help="Number of agents")
    p.add_argument("--episodes", type=int, default=3, help="Episodes per map")
    p.add_argument("--steps", type=int, default=500, help="Max steps per episode")
    p.add_argument("--seed", type=int, default=42, help="Base random seed")
    p.add_argument(
        "--maps",
        nargs="*",
        default=TF_MAPS,
        help="Map filenames from packages/cogames/src/cogames/maps/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    overall = 0.0
    per_map_results: list[tuple[str, float]] = []

    for map_name in args.maps:
        try:
            env_cfg = make_game(num_cogs=args.cogs, map_name=map_name)
            env = MettaGridEnv(env_cfg=env_cfg)
        except Exception as e:
            print(f"SKIP {map_name}: failed to create env ({e})")
            continue

        per_map_sum = 0.0
        for e in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + e)
            policy = ScriptedAgentPolicy(env)
            agents = [policy.agent_policy(i) for i in range(env.num_agents)]

            totals = np.zeros(env.num_agents, dtype=float)
            for _ in range(args.steps):
                actions = np.zeros(env.num_agents, dtype=dtype_actions)
                for i in range(env.num_agents):
                    actions[i] = int(agents[i].step(obs[i]))
                obs, rewards, done, truncated, _ = env.step(actions)
                totals += rewards
                if all(done) or all(truncated):
                    break

            per_map_sum += float(totals.sum())

        env.close()
        avg_sum = per_map_sum / args.episodes if args.episodes > 0 else 0.0
        per_map_results.append((map_name, avg_sum))
        overall += avg_sum
        print(f"{map_name}: avg_sum_reward={avg_sum:.2f}")

    if not per_map_results:
        print("No maps evaluated.")
        return

    aggregate = overall / len(per_map_results)
    print("\n=== Aggregate ===")
    for name, score in per_map_results:
        print(f"{name}: {score:.2f}")
    print(f"OVERALL (avg over maps): {aggregate:.2f}")


if __name__ == "__main__":
    main()
