import numpy as np
import pytest
from omegaconf import OmegaConf

from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.util.hydra import get_cfg
from tests.actions import generate_valid_random_actions


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

    # Calculate performance metrics
    ops_kilo = benchmark.stats["ops"]
    env_steps_per_second = ops_kilo * 1000.0
    agent_steps_per_second = env_steps_per_second * environment.num_agents

    print("\nStep Performance Results:")
    print(f"Environment Kilo OPS: {ops_kilo:.2f} Kops/s")
    print(f"Environment steps per second: {env_steps_per_second:.2f} ops/s")
    print(f"Agent steps per second: {agent_steps_per_second:.2f} ops/s")
    print(f"Total steps executed: {step_count}")
    print(f"Resets triggered: {reset_count}")
    print(f"Reset rate: {reset_count / step_count * 100:.2f}%")

    # Add metrics to benchmark report
    benchmark.extra_info.update(
        {
            "num_agents": environment.num_agents,
            "env_steps_per_second": env_steps_per_second,
            "agent_steps_per_second": agent_steps_per_second,
            "total_steps": step_count,
            "resets_triggered": reset_count,
            "reset_percentage": reset_count / step_count * 100,
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
    # Generate enough actions for all iterations across all rounds
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

    # Calculate and report performance metrics
    ops_kilo = benchmark.stats["ops"]
    env_steps_per_second = ops_kilo * 1000.0
    agent_steps_per_second = env_steps_per_second * env.num_agents

    print("\nPure Step Performance Results:")
    print(f"Environment Kilo OPS: {ops_kilo:.2f} Kops/s")
    print(f"Environment steps per second: {env_steps_per_second:.2f} ops/s")
    print(f"Number of agents: {env.num_agents}")
    print(f"Agent steps per second: {agent_steps_per_second:.2f} ops/s")

    # Add metrics to benchmark report
    benchmark.extra_info.update(
        {
            "num_agents": env.num_agents,
            "env_steps_per_second": env_steps_per_second,
            "agent_steps_per_second": agent_steps_per_second,
            "test_type": "no_reset",
        }
    )


def test_reset_performance(benchmark, environment):
    """
    Benchmark environment reset performance.
    Measures time to reset the environment to initial state.
    """
    env = environment

    reset_count = 0

    def run_reset():
        nonlocal reset_count
        env.reset()
        reset_count += 1

    # Run the benchmark
    benchmark.pedantic(
        run_reset,
        iterations=1,  # One reset per round
        rounds=20,  # Number of rounds
        warmup_rounds=0,  # Warmup rounds to discard
    )

    # Calculate performance metrics
    ops_kilo = benchmark.stats["ops"]
    resets_per_second = ops_kilo * 1000.0

    print("\nReset Performance Results:")
    print(f"Environment Kilo OPS: {ops_kilo:.2f} Kops/s")
    print(f"Resets per second: {resets_per_second:.2f} ops/s")
    print(f"Average reset time: {1 / resets_per_second * 1000:.2f} ms")
    print(f"Total resets executed: {reset_count}")

    # Add metrics to benchmark report
    benchmark.extra_info.update(
        {
            "resets_per_second": resets_per_second,
            "avg_reset_time_ms": 1 / resets_per_second * 1000,
            "total_resets": reset_count,
        }
    )


def test_action_generation_performance(benchmark, environment):
    """
    Benchmark the performance of valid action generation.
    This helps identify if action generation is a bottleneck.
    Uses deterministically random generation.
    """
    env = environment
    num_agents = env.num_agents

    # Set deterministic seed for this test
    np.random.seed(123)

    action_sets_generated = 0

    def run_action_generation():
        nonlocal action_sets_generated
        actions = generate_valid_random_actions(
            env,
            num_agents=num_agents,
            seed=None,  # Use current numpy random state for deterministic sequence
        )
        action_sets_generated += 1
        return actions

    # Run the benchmark
    benchmark.pedantic(
        run_action_generation,
        iterations=1000,  # Action sets per round
        rounds=10,  # Number of rounds
        warmup_rounds=2,  # Warmup rounds
    )

    # Calculate performance metrics
    ops_kilo = benchmark.stats["ops"]
    action_sets_per_second = ops_kilo * 1000.0
    actions_per_second = action_sets_per_second * num_agents

    print("\nAction Generation Performance Results:")
    print(f"Action sets Kilo OPS: {ops_kilo:.2f} Kops/s")
    print(f"Action sets per second: {action_sets_per_second:.2f} ops/s")
    print(f"Individual actions per second: {actions_per_second:.2f} ops/s")
    print(f"Total action sets generated: {action_sets_generated}")

    # Add metrics to benchmark report
    benchmark.extra_info.update(
        {
            "num_agents": num_agents,
            "action_sets_per_second": action_sets_per_second,
            "actions_per_second": actions_per_second,
            "total_action_sets": action_sets_generated,
        }
    )


def test_full_step_cycle_performance(benchmark, environment):
    """
    Benchmark the complete step cycle: action generation + step + reset handling.
    This gives the most realistic performance measurement for actual usage.
    Uses deterministically random actions.
    """
    env = environment
    num_agents = env.num_agents

    # Set deterministic seed for this test
    np.random.seed(456)

    # Performance tracking
    total_steps = 0
    total_resets = 0
    total_action_generations = 0

    def run_full_cycle():
        nonlocal total_steps, total_resets, total_action_generations

        # Generate valid actions (deterministically random)
        actions = generate_valid_random_actions(env, num_agents=num_agents, seed=None)
        total_action_generations += 1

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
    full_cycles_per_second = ops_kilo * 1000.0
    env_steps_per_second = full_cycles_per_second  # 1:1 ratio
    agent_steps_per_second = env_steps_per_second * num_agents

    print("\nFull Cycle Performance Results:")
    print(f"Full cycles Kilo OPS: {ops_kilo:.2f} Kops/s")
    print(f"Full cycles per second: {full_cycles_per_second:.2f} ops/s")
    print(f"Environment steps per second: {env_steps_per_second:.2f} ops/s")
    print(f"Agent steps per second: {agent_steps_per_second:.2f} ops/s")
    print(f"Total steps: {total_steps}")
    print(f"Total resets: {total_resets}")
    print(f"Reset rate: {total_resets / total_steps * 100:.2f}%")

    # Add comprehensive metrics to benchmark report
    benchmark.extra_info.update(
        {
            "num_agents": num_agents,
            "full_cycles_per_second": full_cycles_per_second,
            "env_steps_per_second": env_steps_per_second,
            "agent_steps_per_second": agent_steps_per_second,
            "total_steps": total_steps,
            "total_resets": total_resets,
            "reset_percentage": total_resets / total_steps * 100,
            "test_type": "full_cycle",
        }
    )


@pytest.mark.parametrize("action_type", [None, 0, 1, 2])
def test_step_performance_by_action_type(benchmark, environment, action_type):
    """
    Benchmark step performance for specific action types.
    Useful for identifying if certain actions are more expensive than others.
    Uses deterministically random actions of the specified type.
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

    # Calculate performance metrics
    ops_kilo = benchmark.stats["ops"]
    env_steps_per_second = ops_kilo * 1000.0

    action_type_name = f"action_type_{action_type}" if action_type is not None else "random_actions"

    print(f"\nPerformance for {action_type_name}:")
    print(f"Environment steps per second: {env_steps_per_second:.2f} ops/s")
    print(f"Steps executed: {step_count}")
    print(f"Resets: {reset_count}")

    # Add metrics to benchmark report
    benchmark.extra_info.update(
        {
            "action_type": action_type_name,
            "env_steps_per_second": env_steps_per_second,
            "steps_executed": step_count,
            "resets_triggered": reset_count,
        }
    )
