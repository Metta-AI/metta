#!/usr/bin/env -S uv run python

import argparse
import sys
import time
from typing import List

import numpy as np

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    GameConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    WallConfig,
)
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.map_builder.random_map import RandomMapBuilder
from mettagrid.simulator import Simulator


def create_env(num_agents: int = 20, map_size: int = 40, seed: int = 42):
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=num_agents,
            max_steps=0,
            obs=ObsConfig(width=11, height=11, num_tokens=200),
            actions=ActionsConfig(
                noop=NoopActionConfig(enabled=True),
                move=MoveActionConfig(
                    enabled=True,
                    allowed_directions=[
                        "north",
                        "south",
                        "east",
                        "west",
                        "northeast",
                        "northwest",
                        "southeast",
                        "southwest",
                    ],
                ),
            ),
            objects={
                "wall": WallConfig(render_symbol="⬛"),
            },
            map_builder=RandomMapBuilder.Config(
                width=map_size,
                height=map_size,
                agents=num_agents,
                objects={"wall": 50},
                border_width=1,
                border_object="wall",
                seed=seed,  # Deterministic map
            ),
        )
    )

    simulator = Simulator()
    env = MettaGridPufferEnv(simulator, cfg)
    env.reset()
    return env


def pre_generate_actions(num_agents: int, num_actions: int, total_steps: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, num_actions, size=(total_steps, num_agents))


def run_benchmark_round(env, actions: np.ndarray, start_idx: int, num_steps: int) -> float:
    start = time.perf_counter()
    for i in range(start_idx, start_idx + num_steps):
        env.step(actions[i])
    return time.perf_counter() - start


def calculate_statistics(times: List[float], num_steps: int, num_agents: int) -> dict:
    times_arr = np.array(times)

    # Basic statistics
    mean_time = np.mean(times_arr)
    std_time = np.std(times_arr)
    min_time = np.min(times_arr)
    max_time = np.max(times_arr)

    # Performance metrics
    env_sps_mean = num_steps / mean_time
    env_sps_std = env_sps_mean * (std_time / mean_time) if mean_time > 0 else 0
    agent_sps_mean = env_sps_mean * num_agents
    agent_sps_std = env_sps_std * num_agents

    # Percentiles
    p50 = np.percentile(times_arr, 50)
    p95 = np.percentile(times_arr, 95)
    p99 = np.percentile(times_arr, 99)

    return {
        "mean_time": mean_time,
        "std_time": std_time,
        "min_time": min_time,
        "max_time": max_time,
        "p50_time": p50,
        "p95_time": p95,
        "p99_time": p99,
        "env_sps_mean": env_sps_mean,
        "env_sps_std": env_sps_std,
        "agent_sps_mean": agent_sps_mean,
        "agent_sps_std": agent_sps_std,
        "cv": std_time / mean_time if mean_time > 0 else 0,  # Coefficient of variation
    }


def run_performance(env, iterations: int, rounds: int, warmup: int) -> dict:
    num_agents = env.num_agents
    num_actions = env.single_action_space.n

    total_steps = warmup + (iterations * rounds)
    print(f"Pre-generating {total_steps:,} action sets...")
    actions = pre_generate_actions(num_agents, num_actions, total_steps)

    print(f"Running {warmup:,} warm-up steps...")
    warmup_start = time.perf_counter()
    for i in range(warmup):
        env.step(actions[i])
    warmup_time = time.perf_counter() - warmup_start

    print(f"Running {rounds} rounds of {iterations:,} steps each...")

    round_times = []
    action_idx = warmup

    for round_num in range(rounds):
        round_time = run_benchmark_round(env, actions, action_idx, iterations)
        round_times.append(round_time)
        action_idx += iterations

        if (round_num + 1) % 5 == 0:
            print(f"  Completed round {round_num + 1}/{rounds}")

    stats = calculate_statistics(round_times, iterations, num_agents)

    print("\nConfiguration:")
    print(f"  Agents: {num_agents}")
    print(f"  Iterations: {iterations:,} per round")
    print(f"  Rounds: {rounds} ({stats['mean_time']:.2f}s/round)")
    print(f"  Warm-up: {warmup:,} steps ({warmup_time:.2f}s)")

    print("\nPerformance Metrics:")
    print(f"  Env SPS: {stats['env_sps_mean']:,.0f} ± {stats['env_sps_std']:,.0f}")
    print(f"  Agent SPS: {stats['agent_sps_mean']:,.0f} ± {stats['agent_sps_std']:,.0f}")

    if stats["cv"] < 0.05:
        stability = "Excellent (CV < 5%)"
    elif stats["cv"] < 0.10:
        stability = "Good (CV < 10%)"
    elif stats["cv"] < 0.20:
        stability = "Fair (CV < 20%)"
    else:
        stability = "Poor (CV ≥ 20%) - results may be unreliable"
    print(f"\nStability: {stability}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Improved MettaGrid performance test")
    parser.add_argument("--agents", type=int, default=20, help="Number of agents")
    parser.add_argument("--map-size", type=int, default=40, help="Map width/height")
    parser.add_argument("--iterations", type=int, default=20000, help="Steps per round")
    parser.add_argument("--rounds", type=int, default=20, help="Number of measurement rounds")
    parser.add_argument("--warmup", type=int, default=150000, help="Warm-up steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"Creating environment: {args.agents} agents on {args.map_size}x{args.map_size} map")
    env = create_env(num_agents=args.agents, map_size=args.map_size, seed=args.seed)

    stats = run_performance(env, iterations=args.iterations, rounds=args.rounds, warmup=args.warmup)

    # Return non-zero exit code if performance is unstable
    if stats["cv"] > 0.20:
        print("\nPerformance measurement unstable!")
        sys.exit(1)


if __name__ == "__main__":
    main()
