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
from cogames.policy.nim_agents.agents import ThinkyAgentsMultiPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Simulator
from mettagrid.simulator.rollout import Rollout


def make_machina_open_world_cfg(num_cogs: int = 4):
  mission = None
  for m in MISSIONS:
    if m.full_name() == "machina_1.open_world":
      mission = m
      break
  if mission is None:
    raise RuntimeError("machina_1.open_world mission not found")
  mission = mission.with_variants([NumCogsVariant(num_cogs=num_cogs)])
  return mission.make_env()


def run_thinky(timesteps: int, seed: int = 0):
  cfg = make_machina_open_world_cfg()
  pei = PolicyEnvInterface.from_mg_cfg(cfg)
  multi_policy = ThinkyAgentsMultiPolicy(pei)
  agent_policies = [multi_policy.agent_policy(i) for i in range(pei.num_agents)]

  simulator = Simulator()
  sim = simulator.new_simulation(cfg, seed)
  agents = sim.agents()

  t0 = time.time()
  steps = 0
  restarts = 0
  while steps < timesteps:
    if sim.is_done():
      sim.close()
      restarts += 1
      sim = simulator.new_simulation(cfg, seed + restarts)
      agents = sim.agents()
      continue

    for i, policy in enumerate(agent_policies):
      action = policy.step(agents[i].observation)
      agents[i].set_action(action)

    sim.step()
    steps += 1

  elapsed = time.time() - t0
  print(f"thinky rollout: {timesteps} steps in {elapsed:.3f}s ({timesteps/elapsed:.1f} steps/s)")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--timesteps", type=int, default=100_000)
  args = parser.parse_args()
  run_thinky(args.timesteps)


if __name__ == "__main__":
  main()
