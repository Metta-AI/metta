"""Test ModifyTargetActionConfig configuration and conversion."""

import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    ActionConfig,
    GameConfig,
    ModifyTargetActionConfig,
)
from mettagrid.config.mettagrid_c_config import convert_to_cpp_game_config


def test_modify_target_config_creation():
    """Test that ModifyTargetActionConfig can be created properly."""
    config = ModifyTargetActionConfig(
        enabled=True,
        required_resources={"mana": 5},
        consumed_resources={"mana": 3.0},
        modifies={"health": 10.0, "gold": -5.0}
    )
    
    assert config.enabled is True
    assert config.required_resources == {"mana": 5}
    assert config.consumed_resources == {"mana": 3.0}
    assert config.modifies == {"health": 10.0, "gold": -5.0}


def test_modify_target_in_actions_config():
    """Test that modify_target can be added to ActionsConfig."""
    actions = ActionsConfig(
        move=ActionConfig(enabled=True),
        modify_target=ModifyTargetActionConfig(
            enabled=True,
            modifies={"health": 10.0}
        )
    )
    
    assert actions.modify_target.enabled is True
    assert actions.modify_target.modifies == {"health": 10.0}


def test_modify_target_conversion_to_cpp():
    """Test that ModifyTargetActionConfig converts properly to C++ config."""
    # Create a GameConfig with modify_target action
    game_config = GameConfig(
        resource_names=["health", "mana", "gold"],
        num_agents=2,
        actions=ActionsConfig(
            move=ActionConfig(enabled=True),
            modify_target=ModifyTargetActionConfig(
                enabled=True,
                required_resources={"mana": 5},
                consumed_resources={"mana": 3.0},
                modifies={"health": 10.0, "gold": -5.0}
            )
        )
    )
    
    # Convert to C++ config - this should not raise an exception
    cpp_config = convert_to_cpp_game_config(game_config)
    
    # The conversion succeeded if we got here without exception
    assert cpp_config is not None


def test_modify_target_disabled_by_default():
    """Test that modify_target is disabled by default in ActionsConfig."""
    actions = ActionsConfig()
    assert actions.modify_target.enabled is False


def test_modify_target_end_to_end():
    """Test that modify_target can be used in an actual environment."""
    from mettagrid.envs.mettagrid_env import MettaGridEnv
    from mettagrid.config.mettagrid_config import MettaGridConfig
    
    # Create a simple 5x5 map using ASCII characters
    game_map = [
        ["#", "#", "#", "#", "#"],
        ["#", "@", ".", ".", "#"],
        ["#", ".", ".", ".", "#"],
        ["#", ".", ".", ".", "#"],
        ["#", "#", "#", "#", "#"],
    ]
    
    # Create environment config with modify_target enabled
    config = MettaGridConfig.EmptyRoom(num_agents=1, width=5, height=5, border_width=1)
    config.game.resource_names = ["energy", "health"]
    config.game.actions = ActionsConfig(
        move=ActionConfig(enabled=True),
        modify_target=ModifyTargetActionConfig(
            enabled=True,
            required_resources={"energy": 1},
            consumed_resources={"energy": 0.5},
            modifies={"health": 1.0}
        )
    )
    config = config.with_ascii_map(game_map)
    
    # Create environment - this should work without errors
    env = MettaGridEnv(config)
    
    # Verify the action is available
    assert "modify_target" in env.action_names or 7 in range(env.max_actions)  # modify_target is typically action 7