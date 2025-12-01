"""
Tests for PettingZoo integration with MettaGrid.

This module tests the MettaGridPettingZooEnv with PettingZoo's ParallelEnv interface.
"""

import numpy as np
from pettingzoo.test import parallel_api_test

from mettagrid.builder.envs import make_arena
from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    GameConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    WallConfig,
)
from mettagrid.envs.pettingzoo_env import MettaGridPettingZooEnv
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.mapgen.utils.ascii_grid import DEFAULT_CHAR_TO_NAME
from mettagrid.simulator import Simulator


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
            actions=ActionsConfig(move=MoveActionConfig(), noop=NoopActionConfig()),
            objects={"wall": WallConfig()},
            agents=agents,
            map_builder=AsciiMapBuilder.Config(map_data=map_data, char_to_map_name=DEFAULT_CHAR_TO_NAME),
        )
    )
    return cfg


def test_pettingzoo_env_creation():
    """Test PettingZoo environment creation and properties."""
    cfg = make_pettingzoo_env(num_agents=3, max_steps=100)
    simulator = Simulator()
    env = MettaGridPettingZooEnv(simulator, cfg)

    try:
        # Test environment properties (agent IDs are integers)
        assert env.possible_agents == [0, 1, 2]
        assert env.max_num_agents == 3
        assert env.observation_space(0) is not None
        assert env.action_space(0) is not None
        assert env._cfg.game.max_steps == 100
    finally:
        env.close()
        simulator.close()


def test_pettingzoo_env_reset():
    """Test PettingZoo environment reset functionality."""
    cfg = make_pettingzoo_env(num_agents=3, max_steps=100)
    simulator = Simulator()
    env = MettaGridPettingZooEnv(simulator, cfg)

    try:
        # Test reset
        observations, infos = env.reset(seed=42)

        # Check that we get observations and infos for all agents (agent IDs are integers)
        assert len(observations) == 3
        assert len(infos) == 3
        assert env.agents == [0, 1, 2]

        # Check observation shapes
        for agent_id in env.agents:
            assert agent_id in observations
            assert observations[agent_id].shape == (200, 3)
            assert agent_id in infos
            assert isinstance(infos[agent_id], dict)
    finally:
        env.close()
        simulator.close()


def test_pettingzoo_env_step():
    """Test PettingZoo environment step functionality."""
    cfg = make_pettingzoo_env(num_agents=3)
    simulator = Simulator()
    env = MettaGridPettingZooEnv(simulator, cfg)

    try:
        observations, infos = env.reset(seed=42)

        # Test a few steps
        for _ in range(5):
            # Random actions for active agents (agent IDs are integers)
            actions = {}
            for agent_id in env.agents:
                actions[agent_id] = int(env.action_space(agent_id).sample())

            observations, rewards, terminations, truncations, infos = env.step(actions)

            # Check return types and shapes
            assert isinstance(observations, dict)
            assert isinstance(rewards, dict)
            assert isinstance(terminations, dict)
            assert isinstance(truncations, dict)
            assert isinstance(infos, dict)

            # Check that all active agents have entries
            for agent_id in env.agents:
                assert agent_id in observations
                assert agent_id in rewards
                assert agent_id in terminations
                assert agent_id in truncations
                assert agent_id in infos

                assert observations[agent_id].shape == (200, 3)
                assert isinstance(rewards[agent_id], (int, float))
                assert isinstance(terminations[agent_id], bool)
                assert isinstance(truncations[agent_id], bool)
                assert isinstance(infos[agent_id], dict)
    finally:
        env.close()
        simulator.close()


def test_pettingzoo_env_agent_removal():
    """Test that agents are properly removed when terminated."""
    cfg = make_pettingzoo_env(num_agents=3)
    simulator = Simulator()
    env = MettaGridPettingZooEnv(simulator, cfg)

    try:
        observations, infos = env.reset(seed=42)
        initial_agents = env.agents.copy()

        # Run until some agents might be removed or max steps
        max_test_steps = 50
        step_count = 0

        while env.agents and step_count < max_test_steps:
            actions = {}
            for agent_id in env.agents:
                actions[agent_id] = int(env.action_space(agent_id).sample())

            observations, rewards, terminations, truncations, infos = env.step(actions)
            step_count += 1

            # Check that agent list is consistent with termination/truncation
            for agent_id in initial_agents:
                if agent_id not in env.agents:
                    # Agent should have been terminated or truncated
                    assert terminations.get(agent_id, False) or truncations.get(agent_id, False)
    finally:
        env.close()
        simulator.close()


def test_pettingzoo_env_spaces():
    """Test PettingZoo environment observation and action spaces."""
    cfg = make_pettingzoo_env(num_agents=3)
    simulator = Simulator()
    env = MettaGridPettingZooEnv(simulator, cfg)

    try:
        # Test space methods (agent IDs are integers)
        for agent_id in env.possible_agents:
            obs_space = env.observation_space(agent_id)
            action_space = env.action_space(agent_id)

            assert obs_space is not None
            assert action_space is not None

            # Test that same space objects are returned (PettingZoo requirement)
            assert env.observation_space(agent_id) is obs_space
            assert env.action_space(agent_id) is action_space
    finally:
        env.close()
        simulator.close()


def test_pettingzoo_env_state():
    """Test PettingZoo environment state functionality."""
    cfg = make_pettingzoo_env(num_agents=3)
    simulator = Simulator()
    env = MettaGridPettingZooEnv(simulator, cfg)

    try:
        observations, infos = env.reset(seed=42)

        # Test state and state_space
        state = env.state()
        state_space = env.state_space

        assert state is not None
        assert isinstance(state, np.ndarray)
        assert state_space is not None
        assert state.shape == state_space.shape
    finally:
        env.close()
        simulator.close()


def test_pettingzoo_api_compliance():
    """Test official PettingZoo API compliance."""
    cfg = make_pettingzoo_env(num_agents=3)
    simulator = Simulator()
    env = MettaGridPettingZooEnv(simulator, cfg)

    try:
        # Run the official PettingZoo parallel API compliance test
        parallel_api_test(env, num_cycles=3)
    finally:
        env.close()
        simulator.close()


def test_pettingzoo_episode_lifecycle():
    """Test the complete episode lifecycle with PettingZoo API."""
    cfg = make_pettingzoo_env(num_agents=3)
    simulator = Simulator()
    env = MettaGridPettingZooEnv(simulator, cfg)

    try:
        # Reset environment
        observations, infos = env.reset(seed=42)

        # Check reset return format
        assert isinstance(observations, dict)
        assert isinstance(infos, dict)
        assert len(observations) == len(env.agents)
        assert len(infos) == len(env.agents)

        # Test that all agents are initially active (agent IDs are integers)
        assert len(env.agents) == 3
        assert env.agents == env.possible_agents

        # Run a few steps
        for _step in range(5):
            # Generate random actions for all active agents
            actions = {}
            for agent_id in env.agents:
                action_space = env.action_space(agent_id)
                actions[agent_id] = action_space.sample()

            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # Check step return format
            assert isinstance(observations, dict)
            assert isinstance(rewards, dict)
            assert isinstance(terminations, dict)
            assert isinstance(truncations, dict)
            assert isinstance(infos, dict)

            # Check that all active agents are represented
            for agent_id in env.agents:
                assert agent_id in observations
                assert agent_id in rewards
                assert agent_id in terminations
                assert agent_id in truncations
                assert agent_id in infos

            # Check data types
            for agent_id in env.agents:
                assert isinstance(rewards[agent_id], (int, float))
                assert isinstance(terminations[agent_id], bool)
                assert isinstance(truncations[agent_id], bool)
                assert isinstance(infos[agent_id], dict)

            # If all agents are done, break
            if not env.agents:
                break
    finally:
        env.close()
        simulator.close()


def test_pettingzoo_action_observation_spaces():
    """Test that action and observation spaces are properly configured."""
    cfg = make_pettingzoo_env(num_agents=5)
    simulator = Simulator()
    env = MettaGridPettingZooEnv(simulator, cfg)

    try:
        # Reset to ensure environment is initialized
        env.reset(seed=42)

        # Test that all agents have the same spaces (homogeneous agents, agent IDs are integers)
        reference_obs_space = env.observation_space(env.possible_agents[0])
        reference_action_space = env.action_space(env.possible_agents[0])

        for agent_id in env.possible_agents:
            obs_space = env.observation_space(agent_id)
            action_space = env.action_space(agent_id)

            # In our implementation, all agents have the same spaces
            assert obs_space.shape == reference_obs_space.shape
            assert action_space.n == reference_action_space.n

            # Test that spaces can generate valid samples
            obs_sample = obs_space.sample()
            action_sample = action_space.sample()

            assert obs_space.contains(obs_sample)
            assert action_space.contains(action_sample)
    finally:
        env.close()
        simulator.close()


def test_pettingzoo_render_functionality():
    """Test that rendering works with PettingZoo interface."""
    cfg = make_pettingzoo_env(num_agents=5)
    simulator = Simulator()
    env = MettaGridPettingZooEnv(simulator, cfg)

    try:
        # Reset environment
        env.reset(seed=42)

        # Test render method
        render_result = env.render()

        # Should return None (not implemented)
        assert render_result is None
    finally:
        env.close()
        simulator.close()
