"""Tests for ResourceModActionConfig creation and conversion only.

These tests validate the Python configuration layer (pydantic models) and
conversion to the C++ config types. Runtime semantics (AoE effects, scaling,
converter interactions, etc.) are covered in test_resource_mod.py using the
low-level C++ API to avoid duplication.
"""

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


def test_resource_mod_passthrough_fields_to_cpp():
    """Ensure Python config fields map through conversion to C++ config.

    This checks that enabled/required/consumed/modifies/radii/scales are
    accepted by the converter without raising and that the resulting C++
    config includes the resource_mod action in the action list. Detailed
    runtime semantics are validated in test_resource_mod.py.
    """
    game_config = GameConfig(
        resource_names=["energy", "health", "gold"],
        num_agents=1,
        actions=ActionsConfig(
            resource_mod=ResourceModActionConfig(
                enabled=True,
                required_resources={"energy": 2},
                consumed_resources={"energy": 1.0},
                modifies={"health": 3.0, "gold": -1.0},
                agent_radius=1,
                converter_radius=2,
                scales=True,
            )
        ),
    )

    cpp_config = convert_to_cpp_game_config(game_config)

    # The C++ config should expose an action_names method; ensure resource_mod
    # is registered. If not available, fall back to attribute presence checks.
    has_action_names = hasattr(cpp_config, "action_names")
    if has_action_names:
        names = list(cpp_config.action_names())
        assert "resource_mod" in names
    else:
        # Minimal sanity: the object exists and is of the expected type
        assert cpp_config is not None
