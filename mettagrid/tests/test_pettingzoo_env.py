"""
Tests for PettingZoo integration with MettaGrid.

This module tests the MettaGridPettingZooEnv with PettingZoo's ParallelEnv interface.
"""

import numpy as np
import pytest
from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.pettingzoo_env import MettaGridPettingZooEnv


@pytest.fixture
def simple_config():
    """Create a simple navigation configuration."""
    return DictConfig(
        {
            "game": {
                "max_steps": 100,
                "num_agents": 3,
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
                    "agents": 3,
                    "width": 16,
                    "height": 16,
                    "border_width": 1,
                    "objects": {},
                },
            }
        }
    )


def test_pettingzoo_env_creation(simple_config):
    """Test PettingZoo environment creation and properties."""
    curriculum = SingleTaskCurriculum("pettingzoo_test", simple_config)

    env = MettaGridPettingZooEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    # Test environment properties
    assert env.possible_agents == ["agent_0", "agent_1", "agent_2"]
    assert env.max_num_agents == 3
    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.max_steps == 100

    env.close()


def test_pettingzoo_env_reset(simple_config):
    """Test PettingZoo environment reset functionality."""
    curriculum = SingleTaskCurriculum("pettingzoo_reset_test", simple_config)
    env = MettaGridPettingZooEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    # Test reset
    observations, infos = env.reset(seed=42)

    # Check that we get observations and infos for all agents
    assert len(observations) == 3
    assert len(infos) == 3
    assert env.agents == ["agent_0", "agent_1", "agent_2"]

    # Check observation shapes
    for agent in env.agents:
        assert agent in observations
        assert observations[agent].shape == (50, 3)
        assert agent in infos
        assert isinstance(infos[agent], dict)

    env.close()


def test_pettingzoo_env_step(simple_config):
    """Test PettingZoo environment step functionality."""
    curriculum = SingleTaskCurriculum("pettingzoo_step_test", simple_config)
    env = MettaGridPettingZooEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    observations, infos = env.reset(seed=42)

    # Test a few steps
    for _ in range(5):
        # Random actions for active agents
        actions = {}
        for agent in env.agents:
            actions[agent] = np.random.randint(0, 2, size=2, dtype=np.int32)

        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Check return types and shapes
        assert isinstance(observations, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminations, dict)
        assert isinstance(truncations, dict)
        assert isinstance(infos, dict)

        # Check that all active agents have entries
        for agent in env.agents:
            assert agent in observations
            assert agent in rewards
            assert agent in terminations
            assert agent in truncations
            assert agent in infos

            assert observations[agent].shape == (50, 3)
            assert isinstance(rewards[agent], (int, float))
            assert isinstance(terminations[agent], bool)
            assert isinstance(truncations[agent], bool)
            assert isinstance(infos[agent], dict)

    env.close()


def test_pettingzoo_env_agent_removal(simple_config):
    """Test that agents are properly removed when terminated."""
    curriculum = SingleTaskCurriculum("pettingzoo_removal_test", simple_config)
    env = MettaGridPettingZooEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    observations, infos = env.reset(seed=42)
    initial_agents = env.agents.copy()

    # Run until some agents might be removed or max steps
    max_test_steps = 50
    step_count = 0

    while env.agents and step_count < max_test_steps:
        actions = {}
        for agent in env.agents:
            actions[agent] = np.random.randint(0, 2, size=2, dtype=np.int32)

        observations, rewards, terminations, truncations, infos = env.step(actions)
        step_count += 1

        # Check that agent list is consistent with termination/truncation
        for agent in initial_agents:
            if agent not in env.agents:
                # Agent should have been terminated or truncated
                assert terminations.get(agent, False) or truncations.get(agent, False)

    env.close()


def test_pettingzoo_env_spaces(simple_config):
    """Test PettingZoo environment observation and action spaces."""
    curriculum = SingleTaskCurriculum("pettingzoo_spaces_test", simple_config)
    env = MettaGridPettingZooEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    # Test space methods
    for agent in env.possible_agents:
        obs_space = env.observation_space_for_agent(agent)
        action_space = env.action_space_for_agent(agent)

        assert obs_space is not None
        assert action_space is not None
        assert obs_space == env.observation_space
        assert action_space == env.action_space

    env.close()


def test_pettingzoo_env_state(simple_config):
    """Test PettingZoo environment state functionality."""
    curriculum = SingleTaskCurriculum("pettingzoo_state_test", simple_config)
    env = MettaGridPettingZooEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    observations, infos = env.reset(seed=42)

    # Test state and state_space
    state = env.state()
    state_space = env.state_space

    assert state is not None
    assert isinstance(state, np.ndarray)
    assert state_space is not None
    assert state.shape == state_space.shape

    env.close()
