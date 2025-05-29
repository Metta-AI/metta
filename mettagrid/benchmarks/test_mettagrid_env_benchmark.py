import numpy as np
import pytest
from omegaconf import OmegaConf

from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.util.actions import generate_valid_random_actions
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


def test_step_performance_no_reset(benchmark, environment, action_generator):
    """
    Benchmark pure step method performance without reset overhead.

    CRITICAL ASSUMPTION: Episodes last longer than benchmark iterations.
    This test measures raw step performance by avoiding resets during timing.
    Uses deterministically random actions.
    """
    env = environment

    # Perform initial reset (not timed)
    env.reset()

    # Pre-generate a sequence of deterministic actions for consistent timing
    total_iterations = 1000 * 20  # iterations * rounds
    action_sequence = []
    for _ in range(total_iterations):
        action_sequence.append(action_generator())

    iteration_counter = 0

    def run_step():
        """Pure step operation with pre-generated deterministic actions."""
        nonlocal iteration_counter
        actions = action_sequence[iteration_counter % len(action_sequence)]
        iteration_counter += 1

        obs, rewards, terminated, truncated, infos = env.step(actions)
        # Intentionally ignore termination states to measure pure step performance

    # Run the benchmark
    benchmark.pedantic(
        run_step,
        iterations=1000,
        rounds=20,
        warmup_rounds=5,
    )

    # Calculate throughput KPIs from timing
    ops_kilo = benchmark.stats["ops"]
    env_steps_per_second = ops_kilo * 1000.0
    agent_steps_per_second = env_steps_per_second * env.num_agents

    print("\nPure Step Performance Results:")
    print(f"Latency: {benchmark.stats['mean']:.6f} seconds")
    print(f"Environment steps per second: {env_steps_per_second:.2f}")
    print(f"Agent steps per second: {agent_steps_per_second:.2f}")

    # Report: built-in latency + 2 custom KPIs
    benchmark.extra_info.update(
        {
            "env_steps_per_second": env_steps_per_second,
            "agent_steps_per_second": agent_steps_per_second,
        }
    )


def test_reset_performance(benchmark, environment):
    """
    Benchmark environment reset performance as latency.
    Reports: latency (built-in) - how long each reset takes
    """
    env = environment

    def run_reset():
        env.reset()

    # Run the benchmark
    benchmark.pedantic(
        run_reset,
        iterations=1,  # One reset per round
        rounds=20,  # Number of rounds
        warmup_rounds=0,  # Warmup rounds to discard
    )

    print("\nReset Performance Results:")
    print(f"Reset latency: {benchmark.stats['mean']:.6f} seconds")
    print(f"Reset latency: {benchmark.stats['mean'] * 1000:.2f} ms")

    # Only reports built-in latency measure - no custom metrics needed
    # Lower latency = better reset performance


@pytest.mark.parametrize("action_type", [None, 0, 1, 2])
def test_step_performance_by_action_type(benchmark, environment, action_type):
    """
    Benchmark step performance for specific action types.
    Reports: latency (built-in), env_steps_per_second
    """
    env = environment
    num_agents = env.num_agents

    # Set deterministic seed unique to this action type
    base_seed = 789
    type_seed = base_seed + (action_type if action_type is not None else 0)
    np.random.seed(type_seed)

    step_count = 0
    reset_count = 0

    def run_step_with_action_type():
        nonlocal step_count, reset_count

        # Generate deterministically random actions of the specified type
        actions = generate_valid_random_actions(
            env,
            num_agents=num_agents,
            force_action_type=action_type,
            seed=None,  # Use current numpy random state
        )

        obs, rewards, terminated, truncated, infos = env.step(actions)
        step_count += 1

        if np.any(terminated) or np.any(truncated):
            env.reset()
            reset_count += 1

    # Run the benchmark
    benchmark.pedantic(
        run_step_with_action_type,
        iterations=500,
        rounds=5,
        warmup_rounds=1,
    )

    # Calculate throughput KPI from timing
    ops_kilo = benchmark.stats["ops"]
    env_steps_per_second = ops_kilo * 1000.0

    action_type_name = f"action_type_{action_type}" if action_type is not None else "random_actions"

    print(f"\nPerformance for {action_type_name}:")
    print(f"Latency: {benchmark.stats['mean']:.6f} seconds")
    print(f"Environment steps per second: {env_steps_per_second:.2f}")

    # Report: built-in latency + environment throughput
    benchmark.extra_info.update(
        {
            "env_steps_per_second": env_steps_per_second,
        }
    )
