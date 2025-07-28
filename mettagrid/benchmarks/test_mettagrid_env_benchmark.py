import random

import numpy as np
import pytest
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid import MettaGridEnv
from metta.mettagrid.util.actions import generate_valid_random_actions
from metta.mettagrid.util.hydra import get_cfg


@pytest.fixture
def cfg():
    """Create configuration for the environment."""
    return get_cfg("benchmark")


@pytest.fixture
def environment(cfg, num_agents):
    """Create and initialize the environment with specified number of agents."""
    seed = 42  # Or any fixed seed value
    random.seed(seed)
    np.random.seed(seed)

    # Map from num_agents to expected_hash
    grid_hash_map = {
        1: 10198962306018088423,
        2: 14724462956252883691,
        4: 17314270363189457391,
        8: 7658271300011274487,
        16: 4649249633720493321,
    }

    expected_grid_hash = grid_hash_map.get(num_agents)
    if expected_grid_hash is None:
        raise ValueError(f"No expected hash defined for num_agents={num_agents}")

    # Override the number of agents in the configuration
    cfg.game.num_agents = num_agents
    num_rooms = min(num_agents, 4)
    cfg.game.map_builder.num_rooms = num_rooms
    agents_per_room = num_agents // num_rooms
    cfg.game.map_builder.room.agents = agents_per_room
    cfg.game.max_steps = 0  # env lasts forever

    print(f"\nConfiguring environment with {num_agents} agents")
    print(OmegaConf.to_yaml(cfg))

    curriculum = SingleTaskCurriculum("test", task_cfg=cfg)
    env = MettaGridEnv(curriculum, render_mode="human", recursive=False)

    assert env.initial_grid_hash == expected_grid_hash

    env.reset()

    assert env.initial_grid_hash == expected_grid_hash

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


def test_create_env_performance(benchmark, cfg):
    """
    Benchmark environment creation.

    This test measures the time to create a new environment instance
    and perform a reset operation.

    """

    def create_and_reset():
        """Create a new environment and reset it."""
        curriculum = SingleTaskCurriculum("test", task_cfg=cfg)
        env = MettaGridEnv(curriculum, render_mode="human", recursive=False)
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
