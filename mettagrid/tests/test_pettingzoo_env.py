"""
Tests for PettingZoo integration with MettaGrid.

This module tests the MettaGridPettingZooEnv with PettingZoo's ParallelEnv interface.
"""

import numpy as np
import pytest
from omegaconf import DictConfig
from pettingzoo.test import parallel_api_test

from metta.mettagrid.curriculum import single_task
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
    curriculum = single_task("pettingzoo_test", simple_config)

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
    curriculum = single_task("pettingzoo_reset_test", simple_config)
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
    curriculum = single_task("pettingzoo_step_test", simple_config)
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
    curriculum = single_task("pettingzoo_removal_test", simple_config)
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
    curriculum = single_task("pettingzoo_spaces_test", simple_config)
    env = MettaGridPettingZooEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    # Test space methods
    for agent in env.possible_agents:
        obs_space = env.observation_space(agent)
        action_space = env.action_space(agent)

        assert obs_space is not None
        assert action_space is not None

        # Test that same space objects are returned (PettingZoo requirement)
        assert env.observation_space(agent) is obs_space
        assert env.action_space(agent) is action_space

    env.close()


def test_pettingzoo_env_state(simple_config):
    """Test PettingZoo environment state functionality."""
    curriculum = single_task("pettingzoo_state_test", simple_config)
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


def test_pettingzoo_api_compliance(simple_config):
    """Test official PettingZoo API compliance."""
    curriculum = single_task("pettingzoo_compliance_test", simple_config)
    env = MettaGridPettingZooEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    # Run the official PettingZoo parallel API compliance test
    parallel_api_test(env, num_cycles=3)

    env.close()


def test_pettingzoo_episode_lifecycle(simple_config):
    """Test the complete episode lifecycle with PettingZoo API."""
    curriculum = single_task("pettingzoo_lifecycle_test", simple_config)
    env = MettaGridPettingZooEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    # Reset environment
    observations, infos = env.reset(seed=42)

    # Check reset return format
    assert isinstance(observations, dict)
    assert isinstance(infos, dict)
    assert len(observations) == len(env.agents)
    assert len(infos) == len(env.agents)

    # Test that all agents are initially active
    assert len(env.agents) == 3
    assert env.agents == env.possible_agents

    # Run a few steps
    for _step in range(5):
        # Generate random actions for all active agents
        actions = {}
        for agent in env.agents:
            action_space = env.action_space(agent)
            actions[agent] = action_space.sample()

        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Check step return format
        assert isinstance(observations, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminations, dict)
        assert isinstance(truncations, dict)
        assert isinstance(infos, dict)

        # Check that all active agents are represented
        for agent in env.agents:
            assert agent in observations
            assert agent in rewards
            assert agent in terminations
            assert agent in truncations
            assert agent in infos

        # Check data types
        for agent in env.agents:
            assert isinstance(rewards[agent], (int, float))
            assert isinstance(terminations[agent], bool)
            assert isinstance(truncations[agent], bool)
            assert isinstance(infos[agent], dict)

        # If all agents are done, break
        if not env.agents:
            break

    env.close()


def test_pettingzoo_action_observation_spaces(simple_config):
    """Test that action and observation spaces are properly configured."""
    curriculum = single_task("pettingzoo_spaces_validation_test", simple_config)
    env = MettaGridPettingZooEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    # Reset to ensure environment is initialized
    env.reset(seed=42)

    # Test that all agents have the same spaces (homogeneous agents)
    reference_obs_space = env.observation_space(env.possible_agents[0])
    reference_action_space = env.action_space(env.possible_agents[0])

    for agent in env.possible_agents:
        obs_space = env.observation_space(agent)
        action_space = env.action_space(agent)

        # In our implementation, all agents have the same spaces
        assert obs_space.shape == reference_obs_space.shape
        assert action_space.nvec.tolist() == reference_action_space.nvec.tolist()

        # Test that spaces can generate valid samples
        obs_sample = obs_space.sample()
        action_sample = action_space.sample()

        assert obs_space.contains(obs_sample)
        assert action_space.contains(action_sample)

    env.close()


def test_pettingzoo_render_functionality(simple_config):
    """Test that rendering works with PettingZoo interface."""
    curriculum = single_task("pettingzoo_render_test", simple_config)
    env = MettaGridPettingZooEnv(
        curriculum=curriculum,
        render_mode="human",
        is_training=False,
    )

    # Reset environment
    env.reset(seed=42)

    # Test render method
    render_result = env.render()

    # Should return string representation or None
    assert render_result is None or isinstance(render_result, str)

    env.close()
