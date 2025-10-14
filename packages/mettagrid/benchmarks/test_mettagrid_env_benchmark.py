import random

import numpy as np
import pytest

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.envs.mettagrid_env import MettaGridEnv
from mettagrid.map_builder.random import RandomMapBuilder
from mettagrid.test_support.actions import generate_valid_random_actions


@pytest.fixture
def environment(num_agents: int):
    """Create and initialize the environment with specified number of agents."""
    seed = 42  # Or any fixed seed value
    random.seed(seed)
    np.random.seed(seed)

    # Map from num_agents to expected_hash (updated for deterministic RandomMapBuilder)
    grid_hash_map = {
        1: 8758918251456738458,
        2: 5399377357525131219,
        4: 15159704145714964875,
        8: 17168213948652951998,
        16: 15523353553253390979,
    }

    expected_grid_hash = grid_hash_map.get(num_agents)
    if expected_grid_hash is None:
        raise ValueError(f"No expected hash defined for num_agents={num_agents}")

    cfg = MettaGridConfig()

    # Override the number of agents in the configuration
    cfg.game.num_agents = num_agents
    assert isinstance(cfg.game.map_builder, RandomMapBuilder.Config)
    cfg.game.map_builder.agents = num_agents  # RandomMapBuilderConfig uses agents field
    cfg.game.map_builder.seed = seed  # Set the map builder seed for deterministic maps!
    cfg.game.max_steps = 0  # env lasts forever

    print(f"\nConfiguring environment with {num_agents} agents")

    env = MettaGridEnv(cfg)

    # Verify deterministic grid generation
    assert env.initial_grid_hash == expected_grid_hash, (
        f"Grid hash mismatch for {num_agents} agents! Expected: {expected_grid_hash}, Got: {env.initial_grid_hash}"
    )

    env.reset()

    # Verify that reset doesn't change the initial grid hash
    assert env.initial_grid_hash == expected_grid_hash, (
        f"Grid hash changed after reset for {num_agents} agents! "
        f"Expected: {expected_grid_hash}, Got: {env.initial_grid_hash}"
    )

    yield env
    # Cleanup after test
    del env


@pytest.fixture
def action_generator(environment):
    """
    Create a deterministic action generator function.
    Returns a function that generates different valid actions each call,
    but the sequence is deterministic across test runs.
    """
    # Set the global random seed once for deterministic sequences
    np.random.seed(42)

    def generate_actions():
        return generate_valid_random_actions(
            environment,
            num_agents=environment.num_agents,
            seed=None,  # Use current numpy random state
        )

    return generate_actions


@pytest.mark.parametrize("num_agents", [1, 2, 4, 8, 16])
def test_step_performance(benchmark, environment, action_generator, num_agents):
    """
    Benchmark pure step method performance without reset overhead.

    CRITICAL ASSUMPTION: Episodes last longer than benchmark iterations.
    This test measures raw step performance by avoiding resets during timing.
    Uses deterministically random actions.

    Args:
        num_agents: Number of agents to test (parametrized: 1, 2, 4, 8, 16)
    """
    env = environment

    # Perform initial reset (not timed)
    env.reset()

    # Pre-generate a sequence of deterministic actions for consistent timing
    iterations = 1000
    rounds = 20
    total_iterations = iterations * rounds  # iterations * rounds
    action_sequence = []
    for _ in range(total_iterations):
        action_sequence.append(action_generator())

    iteration_counter = 0

    def run_step():
        """Pure step operation with pre-generated deterministic actions."""
        nonlocal iteration_counter
        actions = action_sequence[iteration_counter % len(action_sequence)]
        iteration_counter += 1

        _obs, _rewards, _terminated, _truncated, _infos = env.step(actions)
        # Intentionally ignore termination states to measure pure step performance

    # Run the benchmark
    benchmark.pedantic(
        run_step,
        iterations=iterations,
        rounds=rounds,
        warmup_rounds=0,
    )

    # Calculate throughput KPIs from timing
    ops_kilo = benchmark.stats["ops"]
    env_rate = ops_kilo * 1000.0
    agent_rate = env_rate * env.num_agents

    print(f"\nPure Step Performance Results ({num_agents} agents):")
    print(f"Latency: {benchmark.stats['mean']:.6f} seconds")
    print(f"Environment rate (steps per second): {env_rate:.2f}")
    print(f"Agent rate (steps per second): {agent_rate:.2f}")

    # Report KPIs
    benchmark.extra_info.update(
        {
            "env_rate": env_rate,
            "agent_rate": agent_rate,
        }
    )


def test_create_env_performance(benchmark):
    """
    Benchmark environment creation.

    This test measures the time to create a new environment instance
    and perform a reset operation.

    """

    def create_and_reset():
        """Create a new environment and reset it."""
        env = MettaGridEnv(MettaGridConfig())
        obs = env.reset()
        # Cleanup
        del env
        return obs

    # Run the benchmark
    benchmark.pedantic(
        create_and_reset,
        iterations=10,
        rounds=3,
        warmup_rounds=1,
    )

    # Calculate KPIs
    create_reset_time = benchmark.stats["mean"]
    env_rate = 1.0 / create_reset_time

    print("\nCreate & Reset Performance Results:")
    print(f"Create + Reset time: {create_reset_time:.6f} seconds")
    print(f"Create + Reset operations per second: {env_rate:.2f}")

    # Report KPIs
    benchmark.extra_info.update({"env_rate": env_rate})
