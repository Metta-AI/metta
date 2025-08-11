"""
Tests for Gymnasium integration with MettaGrid.

This module tests the MettaGridGymEnv with Gymnasium's standard environment interface.
"""

import numpy as np
import pytest
from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.gym_env import MettaGridGymEnv, SingleAgentMettaGridGymEnv


@pytest.fixture
def simple_config():
    """Create a simple navigation configuration."""
    return DictConfig(
        {
            "game": {
                "max_steps": 100,
                "num_agents": 2,
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
                    "agents": 2,
                    "width": 16,
                    "height": 16,
                    "border_width": 1,
                    "objects": {},
                },
            }
        }
    )


def test_multi_agent_gym_env(simple_config):
    """Test multi-agent Gymnasium environment."""
    # Create config and curriculum
    curriculum = SingleTaskCurriculum("gym_multi_test", simple_config)

    # Create environment
    env = MettaGridGymEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
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


def test_single_agent_gym_env(simple_config):
    """Test single-agent Gymnasium environment."""
    # Modify config for single agent
    simple_config.game.num_agents = 1
    simple_config.game.map_builder.agents = 1
    curriculum = SingleTaskCurriculum("gym_single_test", simple_config)

    # Create environment
    env = SingleAgentMettaGridGymEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
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
    curriculum = SingleTaskCurriculum("gym_termination_test", simple_config)
    env = MettaGridGymEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
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
