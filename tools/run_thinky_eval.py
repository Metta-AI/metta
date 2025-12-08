#!/usr/bin/env python
"""
Run a single thinky-controlled rollout for benchmarking.

Usage:
  uv run python tools/run_thinky_eval.py --timesteps 100000
"""

import argparse
import time

import numpy as np

from cogames.cogs_vs_clips.mission import NumCogsVariant
from cogames.cogs_vs_clips.missions import MISSIONS
from cogames.policy.nim_agents import agents as nim_agents
from mettagrid.policy.loader import make_policy_env


def make_machina_open_world_env(num_cogs: int = 4):
  mission = None
  for m in MISSIONS:
    if m.full_name() == "machina_1.open_world":
      mission = m
      break
  if mission is None:
    raise RuntimeError("machina_1.open_world mission not found")
  mission = mission.with_variants([NumCogsVariant(num_cogs=num_cogs)])
  return make_policy_env(mission)


def run_thinky(timesteps: int):
  env = make_machina_open_world_env()
  policy = nim_agents.new_thinky_policy(env.policy_environment_config())

  obs, _ = env.reset()
  t0 = time.time()
  steps = 0
  while steps < timesteps:
    action = policy.step(obs)
    obs, _, done, truncated, _ = env.step(action)
    steps += 1
    if done or truncated:
      obs, _ = env.reset()
  elapsed = time.time() - t0
  print(f"thinky rollout: {timesteps} steps in {elapsed:.3f}s ({timesteps/elapsed:.1f} steps/s)")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--timesteps", type=int, default=100_000)
  args = parser.parse_args()
  run_thinky(args.timesteps)


if __name__ == "__main__":
  main()
