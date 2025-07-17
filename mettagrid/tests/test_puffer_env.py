"""
Tests for PufferLib integration with MettaGrid.

This module tests the MettaGridPufferEnv with PufferLib's environment interface.
"""

import numpy as np
import pytest
from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.puffer_env import MettaGridPufferEnv


@pytest.fixture
def simple_config():
    """Create a simple navigation configuration."""
    return DictConfig(
        {
            "game": {
                "max_steps": 100,
                "num_agents": 4,
                "obs_width": 7,
                "obs_height": 7,
                "num_observation_tokens": 50,
                "inventory_item_names": [
                    "ore_red",
                    "ore_blue",
                    "battery_red",
                    "battery_blue",
                    "heart",
                ],
                "groups": {"agent": {"id": 0, "sprite": 0}},
                "agent": {
                    "default_resource_limit": 10,
                    "rewards": {"inventory": {"heart": 1.0}},
                },
                "actions": {
                    "noop": {"enabled": True},
                    "move": {"enabled": True},
                    "rotate": {"enabled": True},
                    "put_items": {"enabled": True},
                    "get_items": {"enabled": True},
                    "attack": {"enabled": True},
                    "swap": {"enabled": True},
                    "change_color": {"enabled": False},
                    "change_glyph": {"enabled": False, "number_of_glyphs": 0},
                },
                "objects": {
                    "wall": {"type_id": 1, "swappable": False},
                },
                "map_builder": {
                    "_target_": "metta.mettagrid.room.random.Random",
                    "agents": 4,
                    "width": 16,
                    "height": 16,
                    "border_width": 1,
                    "objects": {},
                },
            }
        }
    )


def test_puffer_env_creation(simple_config):
    """Test PufferLib environment creation and properties."""
    curriculum = SingleTaskCurriculum("puffer_test", simple_config)

    env = MettaGridPufferEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    # Test environment properties
    assert env.num_agents == 4
    assert env.single_observation_space is not None
    assert env.single_action_space is not None
    assert env.max_steps == 100

    env.close()


def test_puffer_env_reset(simple_config):
    """Test PufferLib environment reset functionality."""
    curriculum = SingleTaskCurriculum("puffer_reset_test", simple_config)
    env = MettaGridPufferEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    # Test reset
    obs, info = env.reset(seed=42)

    # Check observation shape and types
    assert obs.shape == (4, 50, 3)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)

    env.close()


def test_puffer_env_step(simple_config):
    """Test PufferLib environment step functionality."""
    curriculum = SingleTaskCurriculum("puffer_step_test", simple_config)
    env = MettaGridPufferEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    obs, info = env.reset(seed=42)

    # Test a few steps
    for _ in range(5):
        # Random actions for all agents
        actions = np.random.randint(0, 2, size=(env.num_agents, 2), dtype=np.int32)

        obs, rewards, terminals, truncations, info = env.step(actions)

        # Check return types and shapes
        assert obs.shape == (4, 50, 3)
        assert isinstance(obs, np.ndarray)
        assert isinstance(rewards, np.ndarray) or isinstance(rewards, (int, float))
        assert isinstance(terminals, np.ndarray)
        assert isinstance(truncations, np.ndarray)
        assert isinstance(info, dict)

        # Check array shapes if they're arrays
        if isinstance(rewards, np.ndarray):
            assert rewards.shape == (4,)
        if isinstance(terminals, np.ndarray):
            assert terminals.shape == (4,)
        if isinstance(truncations, np.ndarray):
            assert truncations.shape == (4,)

    env.close()


def test_puffer_env_episode_termination(simple_config):
    """Test that PufferLib environment terminates properly."""
    curriculum = SingleTaskCurriculum("puffer_termination_test", simple_config)
    env = MettaGridPufferEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    obs, info = env.reset(seed=42)

    # Run until termination or max steps
    step_count = 0
    max_test_steps = 150  # More than max_steps to test termination

    while step_count < max_test_steps and not env.done:
        actions = np.random.randint(0, 2, size=(env.num_agents, 2), dtype=np.int32)
        obs, rewards, terminals, truncations, info = env.step(actions)
        step_count += 1

        # Check that we don't exceed max_steps
        if step_count >= env.max_steps:
            assert env.done or (isinstance(truncations, np.ndarray) and any(truncations))
            break

    env.close()


def test_puffer_env_buffer_integration(simple_config):
    """Test PufferLib environment buffer operations."""
    curriculum = SingleTaskCurriculum("puffer_buffer_test", simple_config)
    env = MettaGridPufferEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    # Test that buffers are properly initialized
    obs, info = env.reset(seed=42)

    # The environment should have internal buffers set up
    assert hasattr(env, "_core_env")
    assert env._core_env is not None

    # Test that stepping works with buffers
    actions = np.random.randint(0, 2, size=(env.num_agents, 2), dtype=np.int32)
    obs, rewards, terminals, truncations, info = env.step(actions)

    # Should return properly shaped data
    assert obs.shape == (4, 50, 3)

    env.close()


def test_puffer_env_observation_action_spaces(simple_config):
    """Test PufferLib environment observation and action spaces."""
    curriculum = SingleTaskCurriculum("puffer_spaces_test", simple_config)
    env = MettaGridPufferEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    # Test observation space
    obs_space = env.single_observation_space
    assert obs_space is not None
    assert obs_space.shape == (50, 3)

    # Test action space
    action_space = env.single_action_space
    assert action_space is not None

    # Test that spaces are compatible with actual data
    obs, info = env.reset(seed=42)
    actions = np.random.randint(0, 2, size=(env.num_agents, 2), dtype=np.int32)

    # Check that observations fit the space
    for i in range(env.num_agents):
        assert obs_space.contains(obs[i])

    # Check that actions fit the space
    for i in range(env.num_agents):
        assert action_space.contains(actions[i])

    env.close()
