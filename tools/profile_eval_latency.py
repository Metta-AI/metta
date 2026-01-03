#!/usr/bin/env python3
"""
Profile eval system latency breakdown.

Measures:
1. Pure C++ env stepping (numpy random actions, no Python policy)
2. Random policy through Rollout class (measures Python/Rollout overhead)
3. Neural network policy through Rollout (measures NN inference time)
4. Policy resolution time (metta:// URI to local file)

Usage:
    # Basic profiling with random policy only
    uv run python tools/profile_eval_latency.py

    # With neural network policy (requires local .mpt file)
    uv run python tools/profile_eval_latency.py --policy-path /path/to/checkpoint.mpt

    # With metta:// URI (downloads from S3)
    uv run python tools/profile_eval_latency.py --policy-uri metta://policy/some-policy-name
"""

import argparse
import time

import numpy as np

from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission
from mettagrid.policy.loader import PolicyEnvInterface, initialize_or_load_policy
from mettagrid.policy.policy import PolicySpec
from mettagrid.simulator import Simulation
from mettagrid.simulator.rollout import Rollout


def profile_pure_env(env_cfg, num_agents: int, max_steps: int) -> dict:
    """Profile pure C++ environment stepping with numpy random actions.

    This measures the raw environment performance without any Python policy overhead.
    Actions are generated directly via numpy, bypassing the Rollout class entirely.
    """
    sim = Simulation(env_cfg, seed=42)
    np.random.seed(42)
    num_actions = len(sim.action_names)

    t0 = time.perf_counter()
    for _ in range(max_steps):
        actions = np.random.randint(0, num_actions, size=num_agents, dtype=np.int32)
        sim._c_sim.actions()[:] = actions
        sim.step()
        if sim.is_done():
            break
    elapsed = time.perf_counter() - t0
    steps = sim.current_step
    sim.close()

    return {
        "name": "Pure C++ env stepping",
        "description": "Raw env.step() with numpy random actions, no Python policy",
        "steps": steps,
        "elapsed_s": elapsed,
        "env_sps": steps / elapsed,
        "agent_sps": steps * num_agents / elapsed,
    }


def profile_random_policy(env_cfg, env_interface, num_agents: int) -> dict:
    """Profile random policy through Rollout class.

    This measures the overhead of:
    - The Rollout class machinery
    - Calling policy.step() per agent per step
    - Extracting observations into Python dicts
    - Action validation and setting

    The RandomPolicy itself just returns random.randint(), so inference is ~free.
    """
    policy_spec = PolicySpec(class_path="random")

    t0 = time.perf_counter()
    policies = [
        initialize_or_load_policy(env_interface, policy_spec, device_override="cpu").agent_policy(i)
        for i in range(num_agents)
    ]
    load_time = time.perf_counter() - t0

    rollout = Rollout(env_cfg, policies, max_action_time_ms=10000, render_mode="none", seed=42)

    t0 = time.perf_counter()
    rollout.run_until_done()
    elapsed = time.perf_counter() - t0
    steps = rollout._sim.current_step

    return {
        "name": "Random policy (Rollout)",
        "description": "RandomPolicy through Rollout class, measures Python/Rollout overhead",
        "steps": steps,
        "policy_load_s": load_time,
        "elapsed_s": elapsed,
        "env_sps": steps / elapsed,
        "agent_sps": steps * num_agents / elapsed,
    }


def profile_nn_policy(env_cfg, env_interface, num_agents: int, policy_path: str) -> dict:
    """Profile neural network policy through Rollout class.

    This measures the full eval path including:
    - Neural network forward passes (sequential, one per agent per step)
    - All Rollout class overhead
    - Observation extraction and action setting
    """
    nn_spec = PolicySpec(
        class_path="mettagrid.policy.mpt_policy.MptPolicy",
        data_path=policy_path,
        init_kwargs={"checkpoint_uri": policy_path, "device": "cpu", "strict": "True"},
    )

    t0 = time.perf_counter()
    nn_policies = [
        initialize_or_load_policy(env_interface, nn_spec, device_override="cpu").agent_policy(i)
        for i in range(num_agents)
    ]
    load_time = time.perf_counter() - t0

    rollout = Rollout(env_cfg, nn_policies, max_action_time_ms=10000, render_mode="none", seed=42)

    t0 = time.perf_counter()
    rollout.run_until_done()
    elapsed = time.perf_counter() - t0
    steps = rollout._sim.current_step

    return {
        "name": "Neural network policy",
        "description": "MptPolicy through Rollout, sequential CPU inference",
        "steps": steps,
        "policy_load_s": load_time,
        "elapsed_s": elapsed,
        "env_sps": steps / elapsed,
        "agent_sps": steps * num_agents / elapsed,
    }


def profile_policy_resolution(policy_uri: str) -> dict:
    """Profile metta:// URI resolution time.

    This measures the time to:
    - Resolve metta:// URI to S3 path
    - Download policy zip from S3
    - Extract and cache locally
    """
    from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri

    t0 = time.perf_counter()
    spec = policy_spec_from_uri(policy_uri)
    elapsed = time.perf_counter() - t0

    return {
        "name": "Policy URI resolution",
        "description": f"Resolve {policy_uri} to local path (includes S3 download)",
        "elapsed_s": elapsed,
        "resolved_path": spec.data_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile eval system latency")
    parser.add_argument("--policy-path", help="Path to local .mpt checkpoint for NN profiling")
    parser.add_argument("--policy-uri", help="metta:// URI to resolve and profile")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of agents")
    args = parser.parse_args()

    mission = Machina1OpenWorldMission.model_copy(deep=True)
    mission.num_cogs = args.num_agents
    env_cfg = mission.make_env()
    num_agents = env_cfg.game.num_agents
    max_steps = env_cfg.game.max_steps

    print(f"Environment: {max_steps} steps, {num_agents} agents")
    print("=" * 60)

    env_interface = PolicyEnvInterface.from_mg_cfg(env_cfg)
    results = []

    # 1. Pure env stepping
    print("\n[1/4] Profiling pure C++ env stepping...")
    r = profile_pure_env(env_cfg, num_agents, max_steps)
    results.append(r)
    print(f"      {r['elapsed_s']:.2f}s | {r['env_sps']:.0f} env SPS | {r['agent_sps']:.0f} agent SPS")

    # 2. Random policy
    print("\n[2/4] Profiling random policy through Rollout...")
    r = profile_random_policy(env_cfg, env_interface, num_agents)
    results.append(r)
    print(f"      {r['elapsed_s']:.2f}s | {r['env_sps']:.0f} env SPS | {r['agent_sps']:.0f} agent SPS")

    # 3. Policy URI resolution (if provided)
    policy_path = args.policy_path
    if args.policy_uri:
        print(f"\n[3/4] Profiling policy URI resolution ({args.policy_uri})...")
        r = profile_policy_resolution(args.policy_uri)
        results.append(r)
        print(f"      {r['elapsed_s']:.2f}s to resolve")
        policy_path = r["resolved_path"]
    else:
        print("\n[3/4] Skipping policy URI resolution (no --policy-uri)")

    # 4. Neural network policy (if path available)
    if policy_path:
        print(f"\n[4/4] Profiling neural network policy ({policy_path})...")
        r = profile_nn_policy(env_cfg, env_interface, num_agents, policy_path)
        results.append(r)
        print(f"      Load: {r['policy_load_s']:.2f}s")
        print(f"      Run:  {r['elapsed_s']:.2f}s | {r['env_sps']:.0f} env SPS | {r['agent_sps']:.0f} agent SPS")
    else:
        print("\n[4/4] Skipping NN policy (no --policy-path or --policy-uri)")

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY ({max_steps} steps, {num_agents} agents)")
    print("=" * 60)

    pure_env = next((r for r in results if "Pure" in r["name"]), None)
    random_pol = next((r for r in results if "Random" in r["name"]), None)
    nn_pol = next((r for r in results if "Neural" in r["name"]), None)
    uri_res = next((r for r in results if "URI" in r["name"]), None)

    if pure_env:
        print(f"Pure C++ env:        {pure_env['elapsed_s']:>6.1f}s  (baseline)")
    if random_pol:
        overhead = random_pol["elapsed_s"] - pure_env["elapsed_s"] if pure_env else 0
        print(f"Random policy:       {random_pol['elapsed_s']:>6.1f}s  (+{overhead:.1f}s Rollout overhead)")
    if nn_pol:
        nn_overhead = nn_pol["elapsed_s"] - random_pol["elapsed_s"] if random_pol else nn_pol["elapsed_s"]
        print(f"Neural net policy:   {nn_pol['elapsed_s']:>6.1f}s  (+{nn_overhead:.1f}s NN inference)")
        print(f"  - Policy load:     {nn_pol['policy_load_s']:>6.1f}s")
    if uri_res:
        print(f"Policy URI resolve:  {uri_res['elapsed_s']:>6.1f}s  (S3 download)")

    if nn_pol and random_pol:
        print(f"\nNN is {nn_pol['elapsed_s'] / random_pol['elapsed_s']:.1f}x slower than random policy")
        nn_pct = (nn_pol["elapsed_s"] - pure_env["elapsed_s"]) / nn_pol["elapsed_s"] * 100
        print(f"NN inference is {nn_pct:.1f}% of total time")


if __name__ == "__main__":
    main()
