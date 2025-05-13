import random

import numpy as np
import pytest
from omegaconf import OmegaConf

from mettagrid.config.utils import get_cfg
from mettagrid.mettagrid_c import MettaGrid
from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.resolvers import register_resolvers
from mettagrid.tests.utils import generate_valid_random_actions

# Rebuild the NumPy types using the exposed function
np_observations_type = np.dtype(MettaGrid.get_numpy_type_name("observations"))
np_terminals_type = np.dtype(MettaGrid.get_numpy_type_name("terminals"))
np_truncations_type = np.dtype(MettaGrid.get_numpy_type_name("truncations"))
np_rewards_type = np.dtype(MettaGrid.get_numpy_type_name("rewards"))
np_actions_type = np.dtype(MettaGrid.get_numpy_type_name("actions"))
np_masks_type = np.dtype(MettaGrid.get_numpy_type_name("masks"))
np_success_type = np.dtype(MettaGrid.get_numpy_type_name("success"))

# Define a constant seed for deterministic behavior
BENCHMARK_SEED = 54321


@pytest.fixture
def cfg():
    """Create configuration for the environment."""
    register_resolvers()
    return get_cfg("benchmark")


@pytest.fixture
def environment(cfg):
    """Create and initialize the environment with a fixed seed."""
    # Set seeds for all random number generators
    np.random.seed(BENCHMARK_SEED)
    random.seed(BENCHMARK_SEED)

    print(OmegaConf.to_yaml(cfg))
    env = MettaGridEnv(cfg, render_mode="human", recursive=False, seed=BENCHMARK_SEED)
    env.reset(seed=BENCHMARK_SEED)
    yield env
    # Cleanup after test
    del env


@pytest.fixture
def valid_actions(environment):
    """Generate valid actions for all agents to use for benchmarking."""
    num_agents = environment._c_env.num_agents()
    return generate_valid_random_actions(environment, num_agents, seed=BENCHMARK_SEED)


def test_step_performance(benchmark, environment, valid_actions):
    """Benchmark just the step method performance."""

    def run_step():
        obs, rewards, terminated, truncated, infos = environment.step(valid_actions)
        # Check if any episodes terminated or truncated
        if np.any(terminated) or np.any(truncated):
            environment.reset(seed=BENCHMARK_SEED)

    # Run the benchmark
    benchmark.pedantic(
        run_step,
        iterations=500,  # Number of iterations per round
        rounds=3,  # Number of rounds to run
        warmup_rounds=1,  # Number of warmup rounds to discard
    )


def test_get_stats_performance(benchmark, environment, valid_actions):
    """Benchmark just the get_episode_stats method performance."""
    # First perform some steps to have meaningful stats
    for i in range(10):
        # Use a derived seed for each step to maintain determinism
        iter_seed = BENCHMARK_SEED + i + 1
        iter_actions = generate_valid_random_actions(environment, len(valid_actions), seed=iter_seed)

        obs, rewards, terminated, truncated, infos = environment.step(iter_actions)
        if np.any(terminated) or np.any(truncated):
            environment.reset(seed=iter_seed)

    benchmark.pedantic(
        environment._c_env.get_episode_stats,
        iterations=1000,  # Number of iterations per round
        rounds=3,  # Number of rounds to run
        warmup_rounds=1,  # Number of warmup rounds to discard
    )


def test_combined_performance(benchmark, environment, valid_actions):
    """Benchmark combined step and stats methods for comparison."""
    # Use a counter for deterministic seeds across iterations
    step_count = [0]  # Using a list to allow modification inside the nested function

    def run_step_and_stats():
        # Generate deterministic actions using a derived seed
        iter_seed = BENCHMARK_SEED + step_count[0]
        step_count[0] += 1

        iter_actions = generate_valid_random_actions(environment, len(valid_actions), seed=iter_seed)

        obs, rewards, terminated, truncated, infos = environment.step(iter_actions)
        if np.any(terminated) or np.any(truncated):
            environment.reset(seed=iter_seed)
            return environment._c_env.get_episode_stats()
        return {}  # we only return infos at the end of each episode

    # Reset step count before benchmark
    step_count[0] = 0

    benchmark.pedantic(
        run_step_and_stats,
        iterations=500,  # Number of iterations per round
        rounds=3,  # Number of rounds to run
        warmup_rounds=1,  # Number of warmup rounds to discard
    )
