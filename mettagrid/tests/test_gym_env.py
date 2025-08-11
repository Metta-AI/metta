"""
Tests for Gymnasium integration with MettaGrid.

This module tests the MettaGridGymEnv with Gymnasium's standard environment interface.
"""

import numpy as np
import pytest

from metta.mettagrid.gym_env import MettaGridGymEnv, SingleAgentMettaGridGymEnv
from metta.mettagrid.test_support import env_cfg_builder


@pytest.fixture
def simple_config():
    """Create a simple navigation configuration."""
    return (
        env_cfg_builder()
        .with_num_agents(2)
        .with_max_steps(100)
        .with_obs_size(7, 7, num_tokens=50)
        .with_map_size(16, 16)
        .with_seed(42)
        .build()
    )


def test_multi_agent_gym_env(simple_config):
    """Test multi-agent Gymnasium environment."""
    # Create environment with config
    env = MettaGridGymEnv(
        env_config=simple_config,
        render_mode=None,
        single_agent=False,
    )

    # Test environment properties
    assert env.num_agents == 2
    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.max_steps == 100

    # Test reset
    obs, info = env.reset(seed=42)
    assert obs.shape == (2, 50, 3)
    assert isinstance(info, dict)

    # Test a few steps
    for _ in range(5):
        actions = np.random.randint(0, 2, size=(env.num_agents, 2), dtype=np.int32)
        obs, rewards, terminals, truncations, info = env.step(actions)

        assert obs.shape == (2, 50, 3)
        assert isinstance(rewards, np.ndarray) or isinstance(rewards, (int, float))
        assert isinstance(terminals, np.ndarray)
        assert isinstance(truncations, np.ndarray)
        assert isinstance(info, dict)

    env.close()


def test_single_agent_gym_env():
    """Test single-agent Gymnasium environment."""
    # Create single-agent config
    single_agent_config = (
        env_cfg_builder()
        .with_num_agents(1)
        .with_max_steps(100)
        .with_obs_size(7, 7, num_tokens=50)
        .with_map_size(16, 16)
        .with_seed(42)
        .build()
    )
    # Create environment
    env = SingleAgentMettaGridGymEnv(
        env_config=single_agent_config,
        render_mode=None,
    )

    # Test environment properties
    assert env.num_agents == 1
    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.max_steps == 100

    # Test reset
    obs, info = env.reset(seed=42)
    assert obs.shape == (50, 3)
    assert isinstance(info, dict)

    # Test a few steps
    for _ in range(5):
        action = np.random.randint(0, 2, size=2, dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (50, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    env.close()


def test_gym_env_episode_termination(simple_config):
    """Test that environment terminates properly."""
    env = MettaGridGymEnv(
        env_config=simple_config,
        render_mode=None,
        single_agent=False,
    )

    env.reset(seed=42)

    # Run until termination or max steps
    step_count = 0
    max_test_steps = 150  # More than max_steps to test termination

    while step_count < max_test_steps and not env.done:
        actions = np.random.randint(0, 2, size=(env.num_agents, 2), dtype=np.int32)
        env.step(actions)
        step_count += 1

        # Check that we don't exceed max_steps
        if step_count >= env.max_steps:
            assert env.done
            break

    env.close()
