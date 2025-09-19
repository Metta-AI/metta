"""
Tests for PettingZoo integration with MettaGrid.

This module tests the MettaGridPettingZooEnv with PettingZoo's ParallelEnv interface.
"""

import numpy as np
from pettingzoo.test import parallel_api_test

from mettagrid.builder.envs import make_arena
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    GameConfig,
    MettaGridConfig,
    WallConfig,
)
from mettagrid.envs.pettingzoo_env import MettaGridPettingZooEnv
from mettagrid.map_builder.ascii import AsciiMapBuilder


def make_pettingzoo_env(num_agents=3, max_steps=100):
    """Create a test PettingZoo environment with a simple map."""
    if num_agents == 3:
        map_data = [
            ["#", "#", "#", "#", "#", "#", "#"],
            ["#", ".", ".", ".", ".", ".", "#"],
            ["#", ".", "1", ".", "2", ".", "#"],
            ["#", ".", ".", "3", ".", ".", "#"],
            ["#", ".", ".", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#", "#", "#"],
        ]
    elif num_agents == 5:
        map_data = [
            ["#", "#", "#", "#", "#", "#", "#", "#"],
            ["#", ".", "1", ".", ".", "2", ".", "#"],
            ["#", ".", ".", ".", ".", ".", ".", "#"],
            ["#", ".", "3", ".", ".", "4", ".", "#"],
            ["#", ".", ".", ".", ".", ".", ".", "#"],
            ["#", ".", ".", ".", "@", ".", ".", "#"],
            ["#", ".", ".", ".", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#", "#", "#", "#"],
        ]
    else:
        # Default to num_agents=6 which works with make_arena
        return make_arena(num_agents=num_agents)

    # Create agents with appropriate team IDs
    agents = []
    for i in range(num_agents):
        if i < 4:
            # Numbered agents (1, 2, 3, 4) get team_id based on their number
            agents.append(AgentConfig(team_id=i + 1))
        else:
            # @ agents get team_id 0
            agents.append(AgentConfig(team_id=0))

    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=num_agents,
            max_steps=max_steps,
            actions=ActionsConfig(
                move=ActionConfig(),
                noop=ActionConfig(),
                rotate=ActionConfig(),
            ),
            objects={"wall": WallConfig(type_id=1)},
            agents=agents,
            map_builder=AsciiMapBuilder.Config(map_data=map_data),
        )
    )
    return cfg


def test_pettingzoo_env_creation():
    """Test PettingZoo environment creation and properties."""
    cfg = make_pettingzoo_env(num_agents=3, max_steps=100)
    env = MettaGridPettingZooEnv(
        cfg,
        render_mode=None,
    )

    # Test environment properties
    assert env.possible_agents == ["agent_0", "agent_1", "agent_2"]
    assert env.max_num_agents == 3
    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.max_steps == 100

    env.close()


def test_pettingzoo_env_reset():
    """Test PettingZoo environment reset functionality."""
    cfg = make_pettingzoo_env(num_agents=3, max_steps=100)
    env = MettaGridPettingZooEnv(
        cfg,
        render_mode=None,
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
        assert observations[agent].shape == (200, 3)
        assert agent in infos
        assert isinstance(infos[agent], dict)

    env.close()


def test_pettingzoo_env_step():
    """Test PettingZoo environment step functionality."""
    env = MettaGridPettingZooEnv(
        make_pettingzoo_env(num_agents=3),
        render_mode=None,
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

            assert observations[agent].shape == (200, 3)
            assert isinstance(rewards[agent], (int, float))
            assert isinstance(terminations[agent], bool)
            assert isinstance(truncations[agent], bool)
            assert isinstance(infos[agent], dict)

    env.close()


def test_pettingzoo_env_agent_removal():
    """Test that agents are properly removed when terminated."""
    env = MettaGridPettingZooEnv(
        make_pettingzoo_env(num_agents=3),
        render_mode=None,
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


def test_pettingzoo_env_spaces():
    """Test PettingZoo environment observation and action spaces."""
    env = MettaGridPettingZooEnv(
        make_pettingzoo_env(num_agents=3),
        render_mode=None,
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


def test_pettingzoo_env_state():
    """Test PettingZoo environment state functionality."""
    env = MettaGridPettingZooEnv(
        make_pettingzoo_env(num_agents=3),
        render_mode=None,
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


def test_pettingzoo_api_compliance():
    """Test official PettingZoo API compliance."""
    env = MettaGridPettingZooEnv(
        make_pettingzoo_env(num_agents=3),
        render_mode=None,
    )

    # Run the official PettingZoo parallel API compliance test
    parallel_api_test(env, num_cycles=3)

    env.close()


def test_pettingzoo_episode_lifecycle():
    """Test the complete episode lifecycle with PettingZoo API."""
    env = MettaGridPettingZooEnv(
        make_pettingzoo_env(num_agents=3),
        render_mode=None,
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


def test_pettingzoo_action_observation_spaces():
    """Test that action and observation spaces are properly configured."""
    env = MettaGridPettingZooEnv(
        make_pettingzoo_env(num_agents=5),
        render_mode=None,
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


def test_pettingzoo_render_functionality():
    """Test that rendering works with PettingZoo interface."""
    env = MettaGridPettingZooEnv(
        make_pettingzoo_env(num_agents=5),
        render_mode="human",
    )

    # Reset environment
    env.reset(seed=42)

    # Test render method
    render_result = env.render()

    # Should return string representation or None
    assert render_result is None or isinstance(render_result, str)

    env.close()
