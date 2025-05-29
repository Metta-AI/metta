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


def test_step_performance(benchmark, environment, action_generator):
    """
    Benchmark step method performance with automatic reset handling.
    Uses deterministically random valid actions.
    """
    env = environment

    # Track statistics for reporting
    reset_count = 0
    step_count = 0

    def run_step():
        nonlocal reset_count, step_count

        # Generate new random (but deterministic) actions each step
        actions = action_generator()
        obs, rewards, terminated, truncated, infos = env.step(actions)
        step_count += 1

        # Check if any episodes terminated or truncated
        if np.any(terminated) or np.any(truncated):
            env.reset()
            reset_count += 1

    # Run the benchmark
    _result = benchmark.pedantic(
        run_step,
        iterations=1000,  # Number of iterations per round
        rounds=10,  # Number of rounds to run
        warmup_rounds=2,  # Number of warmup rounds to discard
    )

    # Calculate core KPI metrics
    ops_kilo = benchmark.stats["ops"]
    env_steps_per_second = ops_kilo * 1000.0
    agent_steps_per_second = env_steps_per_second * environment.num_agents
    resets_per_second = (reset_count / step_count) * env_steps_per_second if step_count > 0 else 0

    print("\nStep Performance Results:")
    print(f"Environment steps per second: {env_steps_per_second:.2f}")
    print(f"Agent steps per second: {agent_steps_per_second:.2f}")
    print(f"Resets per second: {resets_per_second:.2f}")

    # Use consistent measure names across all tests
    benchmark.extra_info.update(
        {
            "env_steps_per_second": env_steps_per_second,
            "agent_steps_per_second": agent_steps_per_second,
            "resets_per_second": resets_per_second,
        }
    )


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

    # Run the benchmark with more rounds for better statistics
    benchmark.pedantic(
        run_step,
        iterations=1000,  # Steps per round
        rounds=20,  # Number of rounds
        warmup_rounds=5,  # Warmup rounds to discard
    )

    # Calculate core KPI metrics
    ops_kilo = benchmark.stats["ops"]
    env_steps_per_second = ops_kilo * 1000.0
    agent_steps_per_second = env_steps_per_second * env.num_agents

    print("\nPure Step Performance Results:")
    print(f"Environment steps per second: {env_steps_per_second:.2f}")
    print(f"Agent steps per second: {agent_steps_per_second:.2f}")

    # Same measure names - benchmark name provides context
    benchmark.extra_info.update(
        {
            "env_steps_per_second": env_steps_per_second,
            "agent_steps_per_second": agent_steps_per_second,
        }
    )


def test_reset_performance(benchmark, environment):
    """
    Benchmark environment reset performance.
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

    # Calculate core KPI metrics
    ops_kilo = benchmark.stats["ops"]
    resets_per_second = ops_kilo * 1000.0

    print("\nReset Performance Results:")
    print(f"Resets per second: {resets_per_second:.2f}")

    # Just report throughput metric - no need for redundant latency
    benchmark.extra_info.update(
        {
            "resets_per_second": resets_per_second,
        }
    )


def test_action_generation_performance(benchmark, environment):
    """
    Benchmark the performance of valid action generation.
    """
    env = environment
    num_agents = env.num_agents

    # Set deterministic seed for this test
    np.random.seed(123)

    def run_action_generation():
        actions = generate_valid_random_actions(
            env,
            num_agents=num_agents,
            seed=None,  # Use current numpy random state for deterministic sequence
        )
        return actions

    # Run the benchmark
    benchmark.pedantic(
        run_action_generation,
        iterations=1000,  # Action sets per round
        rounds=10,  # Number of rounds
        warmup_rounds=2,  # Warmup rounds
    )

    # Calculate core KPI metrics
    ops_kilo = benchmark.stats["ops"]
    actions_per_second = ops_kilo * 1000.0 * num_agents  # Total individual actions

    print("\nAction Generation Performance Results:")
    print(f"Actions per second: {actions_per_second:.2f}")

    # Consistent measure names
    benchmark.extra_info.update(
        {
            "actions_per_second": actions_per_second,
        }
    )


def test_full_step_cycle_performance(benchmark, environment):
    """
    Benchmark the complete step cycle: action generation + step + reset handling.
    This gives the most realistic performance measurement for actual usage.
    """
    env = environment
    num_agents = env.num_agents

    # Set deterministic seed for this test
    np.random.seed(456)

    # Performance tracking
    total_steps = 0
    total_resets = 0

    def run_full_cycle():
        nonlocal total_steps, total_resets

        # Generate valid actions (deterministically random)
        actions = generate_valid_random_actions(env, num_agents=num_agents, seed=None)

        # Execute step
        obs, rewards, terminated, truncated, infos = env.step(actions)
        total_steps += 1

        # Handle resets if needed
        if np.any(terminated) or np.any(truncated):
            env.reset()
            total_resets += 1

    # Run the benchmark
    benchmark.pedantic(
        run_full_cycle,
        iterations=500,  # Full cycles per round
        rounds=10,  # Number of rounds
        warmup_rounds=2,  # Warmup rounds
    )

    # Calculate comprehensive performance metrics
    ops_kilo = benchmark.stats["ops"]
    env_steps_per_second = ops_kilo * 1000.0
    agent_steps_per_second = env_steps_per_second * num_agents
    actions_per_second = env_steps_per_second * num_agents  # 1:1 with agent steps
    resets_per_second = (total_resets / total_steps) * env_steps_per_second if total_steps > 0 else 0

    print("\nFull Cycle Performance Results:")
    print(f"Environment steps per second: {env_steps_per_second:.2f}")
    print(f"Agent steps per second: {agent_steps_per_second:.2f}")
    print(f"Actions per second: {actions_per_second:.2f}")
    print(f"Resets per second: {resets_per_second:.2f}")

    # Same measure names - benchmark name provides the context
    benchmark.extra_info.update(
        {
            "env_steps_per_second": env_steps_per_second,
            "agent_steps_per_second": agent_steps_per_second,
            "actions_per_second": actions_per_second,
            "resets_per_second": resets_per_second,
        }
    )


@pytest.mark.parametrize("action_type", [None, 0, 1, 2])
def test_step_performance_by_action_type(benchmark, environment, action_type):
    """
    Benchmark step performance for specific action types.
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

    # Calculate core KPI metrics
    ops_kilo = benchmark.stats["ops"]
    env_steps_per_second = ops_kilo * 1000.0

    action_type_name = f"action_type_{action_type}" if action_type is not None else "random_actions"

    print(f"\nPerformance for {action_type_name}:")
    print(f"Environment steps per second: {env_steps_per_second:.2f}")

    # Consistent measure names across all action type tests
    benchmark.extra_info.update(
        {
            "env_steps_per_second": env_steps_per_second,
        }
    )
