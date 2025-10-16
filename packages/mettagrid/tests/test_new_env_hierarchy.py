"""
Test the new environment hierarchy classes.

This test module verifies that our new MettaGrid environment hierarchy
works correctly while being compatible with the existing test framework.
"""

import numpy as np

from mettagrid import dtype_actions
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    GameConfig,
    MettaGridConfig,
    WallConfig,
)
from mettagrid.envs.mettagrid_env import MettaGridEnv
from mettagrid.envs.pettingzoo_env import MettaGridPettingZooEnv
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.mapgen.utils.ascii_grid import DEFAULT_CHAR_TO_NAME


class TestNewEnvironmentHierarchy:
    """Test new environment hierarchy classes."""

    def test_imports(self):
        """Test that all new classes can be imported."""
        # Test basic imports - this verifies our modules are structured correctly
        assert MettaGridEnv is not None
        assert MettaGridPettingZooEnv is not None

    def test_gym_env_creation(self):
        """Test that Gymnasium environments can be created."""

        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                actions=ActionsConfig(
                    move=ActionConfig(),
                    noop=ActionConfig(),
                    rotate=ActionConfig(),
                ),
                objects={"wall": WallConfig(type_id=1)},
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#", "#", "#"],
                        ["#", ".", ".", ".", "#"],
                        ["#", ".", "@", ".", "#"],
                        ["#", ".", ".", ".", "#"],
                        ["#", "#", "#", "#", "#"],
                    ],
                    char_to_name_map=DEFAULT_CHAR_TO_NAME,
                ),
            )
        )
        env = MettaGridEnv(cfg)

        assert env is not None
        env.close()

    def test_gym_env_basic_ops(self):
        """Test basic Gymnasium environment operations."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                actions=ActionsConfig(
                    move=ActionConfig(),
                    noop=ActionConfig(),
                    rotate=ActionConfig(),
                ),
                objects={"wall": WallConfig(type_id=1)},
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#", "#", "#"],
                        ["#", ".", ".", ".", "#"],
                        ["#", ".", "@", ".", "#"],
                        ["#", ".", ".", ".", "#"],
                        ["#", "#", "#", "#", "#"],
                    ],
                    char_to_name_map=DEFAULT_CHAR_TO_NAME,
                ),
            )
        )
        env = MettaGridEnv(cfg)

        # Test reset
        obs, info = env.reset(seed=42)
        assert obs is not None

        # Test step
        action = np.array([0], dtype=dtype_actions)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert isinstance(reward, np.ndarray)
        assert isinstance(terminated, np.ndarray)
        assert isinstance(truncated, np.ndarray)
        assert isinstance(info, dict)

        env.close()

    def test_pettingzoo_env_creation(self):
        """Test that PettingZoo environment can be created."""
        from mettagrid.envs.pettingzoo_env import MettaGridPettingZooEnv

        # Create PettingZoo config
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=3,
                actions=ActionsConfig(
                    move=ActionConfig(),
                    noop=ActionConfig(),
                    rotate=ActionConfig(),
                ),
                objects={"wall": WallConfig(type_id=1)},
                agents=[
                    AgentConfig(team_id=1),
                    AgentConfig(team_id=2),
                    AgentConfig(team_id=3),
                ],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#", "#", "#", "#", "#"],
                        ["#", ".", ".", ".", ".", ".", "#"],
                        ["#", ".", "1", ".", "2", ".", "#"],
                        ["#", ".", ".", "3", ".", ".", "#"],
                        ["#", ".", ".", ".", ".", ".", "#"],
                        ["#", "#", "#", "#", "#", "#", "#"],
                    ],
                    char_to_name_map=DEFAULT_CHAR_TO_NAME,
                ),
            )
        )
        env = MettaGridPettingZooEnv(cfg, render_mode=None)

        assert env is not None
        env.close()

    def test_pettingzoo_env_basic_ops(self):
        """Test basic PettingZoo environment operations."""
        from mettagrid.envs.pettingzoo_env import MettaGridPettingZooEnv

        # Create multi-agent config
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=3,
                actions=ActionsConfig(
                    move=ActionConfig(),
                    noop=ActionConfig(),
                    rotate=ActionConfig(),
                ),
                objects={"wall": WallConfig(type_id=1)},
                agents=[
                    AgentConfig(team_id=1),
                    AgentConfig(team_id=2),
                    AgentConfig(team_id=3),
                ],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#", "#", "#", "#", "#"],
                        ["#", ".", ".", ".", ".", ".", "#"],
                        ["#", ".", "1", ".", "2", ".", "#"],
                        ["#", ".", ".", "3", ".", ".", "#"],
                        ["#", ".", ".", ".", ".", ".", "#"],
                        ["#", "#", "#", "#", "#", "#", "#"],
                    ],
                    char_to_name_map=DEFAULT_CHAR_TO_NAME,
                ),
            )
        )
        env = MettaGridPettingZooEnv(cfg, render_mode=None)

        # Test reset
        observations, infos = env.reset(seed=42)
        assert isinstance(observations, dict)
        assert isinstance(infos, dict)
        assert len(observations) > 0

        # Test step
        actions = {agent: np.array(0, dtype=dtype_actions) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

        assert isinstance(observations, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminations, dict)
        assert isinstance(truncations, dict)
        assert isinstance(infos, dict)

        env.close()
