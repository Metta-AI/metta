import numpy as np
import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    GameConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
)
from mettagrid.map_builder.random_map import RandomMapBuilder
from mettagrid.simulator import Simulation


@pytest.fixture
def simulation(num_agents: int):
    """Create and initialize the environment with specified number of agents."""
    seed = 42

    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=num_agents,
            obs=ObsConfig(num_tokens=100),
            max_steps=0,  # env lasts forever
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            map_builder=RandomMapBuilder.Config(agents=num_agents, width=20, height=20, seed=seed),
        )
    )

    print(f"\nConfiguring environment with {num_agents} agents")

    sim = Simulation(cfg, seed=seed)

    yield sim
    # Cleanup after test
    sim.close()
    del sim


@pytest.fixture
def action_generator(simulation):
    """Create a deterministic action generator function."""
    np.random.seed(42)

    def generate_actions():
        """Generate random valid actions for all agents."""
        num_actions = len(simulation.action_names)
        return np.random.randint(0, num_actions, size=simulation.num_agents, dtype=np.int32)

    return generate_actions


@pytest.mark.parametrize("num_agents", [1, 2, 4, 8, 16])
def test_step_performance(benchmark, simulation, action_generator, num_agents):
    """Benchmark pure step method performance without reset overhead.

    CRITICAL ASSUMPTION: Episodes last longer than benchmark iterations.
    This test measures raw step performance by avoiding resets during timing.
    Uses deterministically random actions.

    Args:
        num_agents: Number of agents to test (parametrized: 1, 2, 4, 8, 16)
    """
    sim = simulation

    # Pre-generate a sequence of deterministic actions for consistent timing
    iterations = 1000
    rounds = 20
    total_iterations = iterations * rounds
    action_sequence = []
    for _ in range(total_iterations):
        action_sequence.append(action_generator())

    iteration_counter = 0

    def run_step():
        """Pure step operation with pre-generated deterministic actions."""
        nonlocal iteration_counter
        actions = action_sequence[iteration_counter % len(action_sequence)]
        iteration_counter += 1

        # Set actions and step (1D array)
        sim._c_sim.actions()[:] = actions
        sim.step()

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
    agent_rate = env_rate * sim.num_agents

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
    """Benchmark environment creation.

    This test measures the time to create a new Simulation instance.
    """

    def create_env():
        """Create a new environment."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=8,
                actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
                map_builder=RandomMapBuilder.Config(agents=8, width=20, height=20, seed=42),
            )
        )
        sim = Simulation(cfg, seed=42)
        # Cleanup
        sim.close()
        del sim

    # Run the benchmark
    benchmark.pedantic(
        create_env,
        iterations=10,
        rounds=3,
        warmup_rounds=1,
    )

    # Calculate KPIs
    create_time = benchmark.stats["mean"]
    env_rate = 1.0 / create_time

    print("\nCreate Environment Performance Results:")
    print(f"Create time: {create_time:.6f} seconds")
    print(f"Create operations per second: {env_rate:.2f}")

    # Report KPIs
    benchmark.extra_info.update({"env_rate": env_rate})
