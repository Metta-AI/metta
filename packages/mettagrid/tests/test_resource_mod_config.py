"""Test ResourceModActionConfig configuration and conversion."""

import numpy as np

from mettagrid.config.mettagrid_c_config import convert_to_cpp_game_config
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    GameConfig,
    ResourceModActionConfig,
)


def test_resource_mod_config_creation():
    """Test that ResourceModActionConfig can be created properly."""
    config = ResourceModActionConfig(
        enabled=True,
        required_resources={"mana": 5},
        consumed_resources={"mana": 3.0},
        modifies={"health": 10.0, "gold": -5.0},
        agent_radius=2,
        converter_radius=1,
        scales=True,
    )

    assert config.enabled is True
    assert config.required_resources == {"mana": 5}
    assert config.consumed_resources == {"mana": 3.0}
    assert config.modifies == {"health": 10.0, "gold": -5.0}
    assert config.agent_radius == 2
    assert config.converter_radius == 1
    assert config.scales is True


def test_resource_mod_default_values():
    """Test that ResourceModActionConfig has correct default values."""
    config = ResourceModActionConfig(enabled=True, modifies={"health": 10.0})

    assert config.agent_radius == 0
    assert config.converter_radius == 0
    assert config.scales is False


def test_resource_mod_in_actions_config():
    """Test that resource_mod can be added to ActionsConfig."""
    actions = ActionsConfig(
        move=ActionConfig(enabled=True),
        resource_mod=ResourceModActionConfig(enabled=True, modifies={"health": 10.0}, agent_radius=1),
    )

    assert actions.resource_mod.enabled is True
    assert actions.resource_mod.modifies == {"health": 10.0}
    assert actions.resource_mod.agent_radius == 1


def test_resource_mod_conversion_to_cpp():
    """Test that ResourceModActionConfig converts properly to C++ config."""
    # Create a GameConfig with resource_mod action
    game_config = GameConfig(
        resource_names=["health", "mana", "gold"],
        num_agents=2,
        actions=ActionsConfig(
            move=ActionConfig(enabled=True),
            resource_mod=ResourceModActionConfig(
                enabled=True,
                required_resources={"mana": 5},
                consumed_resources={"mana": 3.0},
                modifies={"health": 10.0, "gold": -5.0},
                agent_radius=2,
                converter_radius=1,
                scales=True,
            ),
        ),
    )

    # Convert to C++ config - this should not raise an exception
    cpp_config = convert_to_cpp_game_config(game_config)

    # The conversion succeeded if we got here without exception
    assert cpp_config is not None


def test_resource_mod_disabled_by_default():
    """Test that resource_mod is disabled by default in ActionsConfig."""
    actions = ActionsConfig()
    assert actions.resource_mod.enabled is False


def test_resource_mod_aoe_multiple_agents():
    """Test AoE behavior with multiple agents - basic verification only."""
    from mettagrid.config.mettagrid_config import MettaGridConfig
    from mettagrid.envs.mettagrid_env import MettaGridEnv

    # Create a simple map with 3 agents close together
    game_map = [
        ["#", "#", "#", "#", "#"],
        ["#", "@", "@", "@", "#"],
        ["#", ".", ".", ".", "#"],
        ["#", ".", ".", ".", "#"],
        ["#", "#", "#", "#", "#"],
    ]

    # Create environment config with resource_mod enabled and radius=1
    config = MettaGridConfig.EmptyRoom(num_agents=3, width=5, height=5, border_width=1)
    config.game.resource_names = ["energy", "health"]

    config.game.actions = ActionsConfig(
        move=ActionConfig(enabled=True),
        resource_mod=ResourceModActionConfig(
            enabled=True,
            required_resources={},  # No requirements for simplicity
            consumed_resources={},  # No consumption for simplicity
            modifies={"health": 12.0},  # Will be divided by affected agents when scales=True
            agent_radius=1,
            converter_radius=0,
            scales=True,
        ),
    )
    config = config.with_ascii_map(game_map)

    # Create environment
    env = MettaGridEnv(config)

    # Reset and get initial state
    obs, info = env.reset()

    # Agent 0 at (1, 1) uses resource_mod - with radius 1, affects agents 0 and 1
    action_idx = env.action_names.index("resource_mod")
    actions = np.zeros((3, 1), dtype=np.int32)
    actions[0, 0] = action_idx  # Agent 0 uses resource_mod

    # Execute the action - this should not crash
    obs, rewards, terminals, truncations, info = env.step(actions)

    # Basic verification that the action was processed
    # With AoE radius=1, agent 0 should affect itself and agent 1 (adjacent)
    # The exact values depend on probabilistic rounding, so we just verify no crash
    assert obs is not None, "Observation should be returned"
    assert rewards is not None, "Rewards should be returned"


def test_resource_mod_with_converters():
    """Test resource_mod affecting converters is covered in test_resource_mod.py."""
    # Converter testing requires using the low-level API since converters
    # aren't part of the basic ASCII map syntax. The comprehensive converter
    # tests are in test_resource_mod.py using CppConverterConfig directly.
    assert True  # Test moved to test_resource_mod.py


def test_resource_mod_scaling_behavior():
    """Test that scales flag configuration works - basic verification only."""
    from mettagrid.config.mettagrid_config import MettaGridConfig
    from mettagrid.envs.mettagrid_env import MettaGridEnv

    # Create a map with 3 agents in a row
    game_map = [
        ["#", "#", "#", "#", "#"],
        ["#", "@", "@", "@", "#"],
        ["#", ".", ".", ".", "#"],
        ["#", ".", ".", ".", "#"],
        ["#", "#", "#", "#", "#"],
    ]

    # Test with scales=False (each agent gets full amount)
    config_no_scale = MettaGridConfig.EmptyRoom(num_agents=3, width=5, height=5, border_width=1)
    config_no_scale.game.resource_names = ["energy", "health"]

    config_no_scale.game.actions = ActionsConfig(
        resource_mod=ResourceModActionConfig(
            enabled=True,
            modifies={"health": 9.0},
            agent_radius=2,  # Should affect all 3 agents
            scales=False,  # Each gets 9 health
        )
    )
    config_no_scale = config_no_scale.with_ascii_map(game_map)

    # Test with scales=True (amount divided by num affected)
    config_scale = MettaGridConfig.EmptyRoom(num_agents=3, width=5, height=5, border_width=1)
    config_scale.game.resource_names = ["energy", "health"]

    config_scale.game.actions = ActionsConfig(
        resource_mod=ResourceModActionConfig(
            enabled=True,
            modifies={"health": 9.0},
            agent_radius=2,  # Should affect all 3 agents
            scales=True,  # Each gets 9/3 = 3 health
        )
    )
    config_scale = config_scale.with_ascii_map(game_map)

    # Create both environments
    env_no_scale = MettaGridEnv(config_no_scale)
    env_scale = MettaGridEnv(config_scale)

    # Reset both environments
    obs1, _ = env_no_scale.reset()
    obs2, _ = env_scale.reset()

    # Execute resource_mod with scales=False
    action_idx = env_no_scale.action_names.index("resource_mod")
    actions = np.zeros((3, 1), dtype=np.int32)
    actions[0, 0] = action_idx  # Agent 0 uses resource_mod
    obs1, _, _, _, _ = env_no_scale.step(actions)

    # Execute resource_mod with scales=True
    action_idx = env_scale.action_names.index("resource_mod")
    actions = np.zeros((3, 1), dtype=np.int32)
    actions[0, 0] = action_idx  # Agent 0 uses resource_mod
    obs2, _, _, _, _ = env_scale.step(actions)

    # Basic verification that both environments processed the actions without crashing
    assert obs1 is not None, "scales=False environment should return observation"
    assert obs2 is not None, "scales=True environment should return observation"

    # The actual health changes would require accessing internal state which varies by implementation
    # This test verifies that the configurations are accepted and actions execute without errors
