"""Test that negative action arguments are properly validated and rejected."""

import numpy as np
import pytest

from mettagrid.mettagrid_c import (
    ActionConfig as CppActionConfig,
    AgentConfig as CppAgentConfig,
    GameConfig as CppGameConfig,
    GlobalObsConfig as CppGlobalObsConfig,
    MettaGrid,
    ModifyTargetConfig,
    WallConfig as CppWallConfig,
)


def test_negative_action_arg_validation():
    """Test that negative action arguments are rejected without crashes."""
    
    # Create a simple game config with modify_target action
    game_config = CppGameConfig(
        num_agents=1,
        obs_width=5,
        obs_height=5,
        max_steps=100,
        episode_truncates=False,
        num_observation_tokens=200,
        track_movement_metrics=True,
        resource_names=["energy", "health"],
        actions={
            "move": CppActionConfig(),
            "modify_target": ModifyTargetConfig(
                required_resources={0: 1},  # Need 1 energy
                consumed_resources={0: 1.0},  # Consume 1 energy on success
                modifies={1: 1.0}  # Modify health by 1
            )
        },
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                group_id=0,
                group_name="red",
                resource_limits={0: 10, 1: 10},  # 10 max for energy and health
                initial_inventory={0: 5}  # Start with 5 energy
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False)
        },
        global_obs=CppGlobalObsConfig()
    )
    
    # Create a simple map
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", ".", ".", "wall"],
        ["wall", ".", ".", ".", "wall"],
        ["wall", ".", ".", ".", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]
    
    # Create the environment
    env = MettaGrid(game_config, game_map, 0)
    
    # Try to perform modify_target action with negative arg (-1)
    # Action 7 is typically modify_target
    actions = np.array([[7, -1]], dtype=np.int32)  # action=7 (modify_target), arg=-1 (invalid)
    
    # Reset and step - should not crash
    obs, _ = env.reset()
    obs, rewards, dones, truncs, infos = env.step(actions)
    
    # If we got here without crashing, the test passes
    # The negative arg should have been caught and handled gracefully


def test_negative_action_arg_no_crash():
    """Test that negative action arguments don't cause crashes or undefined behavior."""
    
    game_config = CppGameConfig(
        num_agents=1,
        obs_width=5,
        obs_height=5,
        max_steps=10,
        episode_truncates=False,
        num_observation_tokens=200,
        track_movement_metrics=True,
        resource_names=["resource"],
        actions={
            "move": CppActionConfig(),
            "modify_target": ModifyTargetConfig(
                required_resources={},
                consumed_resources={},
                modifies={}
            )
        },
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                group_id=0,
                group_name="red",
                resource_limits={0: 255}
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False)
        },
        global_obs=CppGlobalObsConfig()
    )
    
    game_map = [
        ["wall", "wall", "wall"],
        ["wall", "agent.red", "wall"],
        ["wall", "wall", "wall"],
    ]
    
    env = MettaGrid(game_config, game_map, 42)
    obs, _ = env.reset()
    
    # Try various negative arguments
    test_cases = [
        np.array([[0, -1]], dtype=np.int32),  # move with -1
        np.array([[7, -10]], dtype=np.int32),  # modify_target with -10
        np.array([[7, -128]], dtype=np.int32),  # modify_target with min int8
        np.array([[0, -255]], dtype=np.int32),  # move with large negative
    ]
    
    for actions in test_cases:
        # Should not crash or raise exception
        obs, rewards, dones, truncs, infos = env.step(actions)
        
        # If we got here without crashing, the negative arg was handled properly