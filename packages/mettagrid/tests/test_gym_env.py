"""
Tests for Gymnasium integration with MettaGrid.

This module tests the MettaGridGymEnv with Gymnasium's standard environment interface.
"""

import numpy as np

from mettagrid.config.mettagrid_config import ActionConfig, ActionsConfig, GameConfig, MettaGridConfig, WallConfig
from mettagrid.envs.gym_env import MettaGridGymEnv
from mettagrid.map_builder.ascii import AsciiMapBuilder


def test_single_agent_gym_env():
    """Test single-agent Gymnasium environment."""
    # Create environment with a simple map
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
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
    env = MettaGridGymEnv(
        cfg,
        render_mode=None,
    )

    # Test environment properties
    assert env.num_agents == 1
    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.max_steps == 100

    # Test reset
    obs, info = env.reset(seed=42)
    assert obs.shape == (200, 3)
    assert isinstance(info, dict)

    # Test a few steps
    for _ in range(5):
        action = np.random.randint(0, 2, size=2, dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (200, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    env.close()


def test_gym_env_episode_termination():
    """Test that environment terminates properly."""
    # Create environment with a simple map
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
    cfg.game.max_steps = 100
    env = MettaGridGymEnv(
        cfg,
        render_mode=None,
    )

    env.reset(seed=42)

    # Run until termination or max steps
    step_count = 0
    max_test_steps = 150  # More than max_steps to test termination

    while step_count < max_test_steps:
        actions = np.random.randint(0, 2, size=(env.num_agents, 2), dtype=np.int32)
        obs, reward, term, trunc, info = env.step(actions)
        step_count += 1

        # Check that we don't exceed max_steps
        if step_count >= env.max_steps:
            assert term or trunc
            break

    env.close()
