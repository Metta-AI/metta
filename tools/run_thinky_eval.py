#!/usr/bin/env python
"""
Run a single thinky-controlled rollout for benchmarking.

Usage:
  uv run python tools/run_thinky_eval.py --timesteps 100000 --seed 0
"""

import argparse
import time

import numpy as np

from cogames.cogs_vs_clips.mission import Mission, NumCogsVariant
from cogames.cogs_vs_clips.missions import MISSIONS
from cogames.policy.nim_agents.agents import ThinkyAgentsMultiPolicy
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Simulator


class _PolicyEnv(MettaGridPufferEnv):
    """Minimal wrapper that exposes policy_environment_config for Nim agents."""

    def __init__(self, mission: Mission, seed: int):
        cfg = mission.make_env()
        self._policy_env_info = PolicyEnvInterface.from_mg_cfg(cfg)
        self._policy_env_json = self._policy_env_info.to_json()
        super().__init__(Simulator(), cfg, seed=seed)

    def policy_environment_config(self) -> str:
        return self._policy_env_json

    @property
    def policy_env_info(self) -> PolicyEnvInterface:
        return self._policy_env_info


def make_machina_open_world_env(num_cogs: int = 4, seed: int = 0) -> _PolicyEnv:
    mission = None
    for m in MISSIONS:
        if m.full_name() == "machina_1.open_world":
            mission = m
            break
    if mission is None:
        raise RuntimeError("machina_1.open_world mission not found")
    mission = mission.with_variants([NumCogsVariant(num_cogs=num_cogs)])
    return _PolicyEnv(mission, seed)


def run_thinky(timesteps: int, seed: int = 0) -> None:
    env = make_machina_open_world_env(seed=seed)
    policy = ThinkyAgentsMultiPolicy(env.policy_env_info)

    actions = np.zeros(env.num_agents, dtype=np.int32)
    next_seed = seed
    obs, _ = env.reset(seed=next_seed)
    next_seed += 1
    t0 = time.time()
    steps = 0
    while steps < timesteps:
        actions.fill(0)
        policy.step_batch(obs, actions)
        obs, _, done, truncated, _ = env.step(actions)
        steps += 1
        if done.any() or truncated.any():
            obs, _ = env.reset(seed=next_seed)
            next_seed += 1
    elapsed = time.time() - t0
    print(f"thinky rollout: {timesteps} steps in {elapsed:.3f}s ({timesteps / elapsed:.1f} steps/s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_thinky(args.timesteps, seed=args.seed)


if __name__ == "__main__":
    main()
