import numpy as np
import pytest
from omegaconf import OmegaConf

from mettagrid.config.utils import get_cfg
from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.resolvers import register_resolvers


@pytest.fixture
def cfg():
    """Create configuration for the environment."""
    register_resolvers()
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

    def run_step():
        obs, rewards, terminated, truncated, infos = environment.step(single_action)
        # Check if any episodes terminated or truncated
        if np.any(terminated) or np.any(truncated):
            environment.reset()

    # Run the benchmark
    benchmark.pedantic(
        run_step,
        iterations=500,  # Number of iterations per round
        rounds=3,  # Number of rounds to run
        warmup_rounds=1,  # Number of warmup rounds to discard
    )


def test_get_stats_performance(benchmark, environment, single_action):
    """Benchmark just the get_episode_stats method performance."""
    # First perform some steps to have meaningful stats
    for _ in range(10):
        obs, rewards, terminated, truncated, infos = environment.step(single_action)
        if np.any(terminated) or np.any(truncated):
            environment.reset()

    benchmark.pedantic(
        environment._c_env.get_episode_stats,
        iterations=1000,  # Number of iterations per round
        rounds=3,  # Number of rounds to run
        warmup_rounds=1,  # Number of warmup rounds to discard
    )


def test_combined_performance(benchmark, environment, single_action):
    """Benchmark combined step and stats methods for comparison."""

    def run_step_and_stats():
        obs, rewards, terminated, truncated, infos = environment.step(single_action)
        if np.any(terminated) or np.any(truncated):
            environment.reset()
            return environment._c_env.get_episode_stats()
        return {}  # we only return infos at the end of each episode

    benchmark.pedantic(
        run_step_and_stats,
        iterations=500,  # Number of iterations per round
        rounds=3,  # Number of rounds to run
        warmup_rounds=1,  # Number of warmup rounds to discard
    )
