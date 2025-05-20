import numpy as np
import pytest
from omegaconf import OmegaConf

from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.util.hydra import get_cfg


@pytest.fixture
def cfg():
    """Create configuration for the environment."""
    return get_cfg("benchmark")


@pytest.fixture
def environment(cfg):
    """Create and initialize the environment."""
    print(OmegaConf.to_yaml(cfg))
    env = MettaGridEnv(cfg, render_mode="human", recursive=False)
    env.reset()
    yield env
    # Cleanup after test
    del env


@pytest.fixture
def single_action(environment):
    """Generate an array of actions with shape (num_agents, 2) to use for benchmarking."""
    return environment.action_space.sample()[0]


def test_step_performance(benchmark, environment, single_action):
    """Benchmark just the step method performance."""

    np.random.seed(42)

    def run_step():
        obs, rewards, terminated, truncated, infos = environment.step(single_action)
        # Check if any episodes terminated or truncated
        if np.any(terminated) or np.any(truncated):
            environment.reset()

    # Run the benchmark
    benchmark.pedantic(
        run_step,
        iterations=1000,  # Number of iterations per round
        rounds=10,  # Number of rounds to run
        warmup_rounds=0,  # Number of warmup rounds to discard
    )


def test_step_performance_no_reset(benchmark, environment, single_action):
    """
    Benchmark just the env.step() method performance.
    This test excludes environment reset time by relying on the CRITICAL ASSUMPTION
    that a single episode will last longer than the number of iterations
    configured for the benchmark (e.g., 1000 steps per round). An initial reset
    is performed before the benchmark, but no resets are handled *during* the
    timed iterations.
    """
    np.random.seed(42)

    env = environment
    # Perform initial reset before benchmarking. This is not timed.
    env.reset()

    # Get the number of agents in the environment
    num_agents = env.num_agents

    def run_step():
        # This function is called repeatedly by pytest-benchmark.
        # It only contains the env.step() call, relying on the assumption
        # that the environment has been reset initially and will not
        # terminate during the benchmark's iterations per round (e.g., 1000 steps).
        # If this assumption is violated, env.step() might be called on a
        # terminated environment, potentially leading to errors or incorrect behavior.
        obs, rewards, terminated, truncated, infos = env.step(single_action)
        # The 'terminated' status from env.step() is intentionally not used here
        # to trigger a reset within the benchmarked function, due to the
        # aforementioned assumption about episode length relative to benchmark iterations.

    # Run the benchmark
    benchmark.pedantic(
        run_step,
        iterations=1000,  # Number of iterations per round
        rounds=20,  # Number of rounds to run
        warmup_rounds=10,  # Number of warmup rounds to discard
    )

    # Calculate and print agent steps per second from benchmark data
    # benchmark.stats['ops'] should give the value from the 'OPS (Kops/s)' column.
    ops_kilo = benchmark.stats["ops"]
    env_steps_per_second = ops_kilo * 1000.0

    agent_steps_per_second = env_steps_per_second * num_agents

    print(f"\nEnvironment Kilo OPS (from stats): {ops_kilo:.2f} Kops/s")
    print(f"Environment steps per second: {env_steps_per_second:.2f} ops/s")
    print(f"Agents: {num_agents}")
    print(f"Agent steps per second: {agent_steps_per_second:.2f} ops/s")

    # Add custom info to the benchmark report
    benchmark.extra_info["num_agents"] = num_agents
    benchmark.extra_info["env_steps_per_second"] = env_steps_per_second
    benchmark.extra_info["agent_steps_per_second"] = agent_steps_per_second


def test_reset_performance(benchmark, environment):
    """
    Benchmark just the env.reset() method performance.
    """
    np.random.seed(42)

    env = environment

    def run_reset():
        # This function is called repeatedly by pytest-benchmark.
        # It only contains the env.reset() call.
        env.reset()

    # Run the benchmark
    benchmark.pedantic(
        run_reset,
        iterations=1,  # Number of iterations per round
        rounds=200,  # Number of rounds to run
        warmup_rounds=50,  # Number of warmup rounds to discard
    )

    # Calculate and print resets per second from benchmark data
    ops_kilo = benchmark.stats["ops"]
    resets_per_second = ops_kilo * 1000.0

    print(f"\nEnvironment Kilo OPS (from stats): {ops_kilo:.2f} Kops/s")
    print(f"Resets per second: {resets_per_second:.2f} ops/s")

    # Add custom info to the benchmark report
    benchmark.extra_info["resets_per_second"] = resets_per_second
