"""
Test the new environment hierarchy classes.

This test module verifies that our new MettaGrid environment hierarchy
works correctly while being compatible with the existing test framework.
"""

import numpy as np

from mettagrid import dtype_actions
from mettagrid.builder.envs import make_arena
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    GameConfig,
    MettaGridConfig,
    WallConfig,
)
from mettagrid.map_builder.ascii import AsciiMapBuilder


class TestNewEnvironmentHierarchy:
    """Test new environment hierarchy classes."""

    def test_imports(self):
        """Test that all new classes can be imported."""
        # Test basic imports - this verifies our modules are structured correctly
        from mettagrid.envs.gym_env import MettaGridGymEnv
        from mettagrid.envs.mettagrid_env import MettaGridEnv
        from mettagrid.envs.pettingzoo_env import MettaGridPettingZooEnv

        assert MettaGridEnv is not None
        assert MettaGridGymEnv is not None
        assert MettaGridPettingZooEnv is not None

    def test_gym_env_creation(self):
        """Test that Gymnasium environments can be created."""
        from mettagrid.envs.gym_env import MettaGridGymEnv

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
                ),
            )
        )
        env = MettaGridGymEnv(cfg, render_mode=None)

        assert env is not None
        env.close()

    def test_gym_env_basic_ops(self):
        """Test basic Gymnasium environment operations."""
        from mettagrid.envs.gym_env import MettaGridGymEnv

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
                ),
            )
        )
        env = MettaGridGymEnv(cfg, render_mode=None)

        # Test reset
        obs, info = env.reset(seed=42)
        assert obs is not None

        # Test step
        action = np.array(0, dtype=dtype_actions)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
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


if __name__ == "__main__":
    # Run tests manually if called directly
    import sys

    print("Running manual tests...")

    # Test imports
    try:
        from mettagrid.envs.mettagrid_env import MettaGridEnv

        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)

    # Test basic creation
    try:
        env = MettaGridEnv(make_arena(num_agents=24), render_mode=None)
        env.close()
        print("✓ MettaGridEnv creation successful")
    except Exception as e:
        print(f"✗ MettaGridEnv creation failed: {e}")
        sys.exit(1)

    print("✓ All manual tests passed!")
