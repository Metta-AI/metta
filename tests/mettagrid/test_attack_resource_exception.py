"""Test that exceptions are raised when attack resources are not in inventory."""

import pytest

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config
from mettagrid.tests.conftest import create_test_config


def test_exception_when_laser_not_in_inventory():
    """Test that an exception is raised when attack requires laser but it's not in inventory_item_names."""
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", ".", "agent.blue", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    # Use create_test_config but exclude laser from inventory to trigger exception
    game_config = create_test_config({
        "max_steps": 50,
        "num_agents": 2,
        "obs_width": 11,
        "obs_height": 11,
        "num_observation_tokens": 200,
        # Note: laser is NOT in inventory_item_names
        "inventory_item_names": ["armor", "heart"],
        "actions": {
            "attack": {
                "enabled": True,
                "consumed_resources": {"laser": 1},  # This should trigger an exception!
                "defense_resources": {"armor": 1},
            },
            "put_items": {"enabled": True},
            "get_items": {"enabled": True},
            "swap": {"enabled": True},
        },
        "groups": {"red": {"id": 0, "props": {}}, "blue": {"id": 1, "props": {}}},
    })

    # Check that creating the environment raises an exception
    with pytest.raises(ValueError) as exc_info:
        # This should trigger an exception about missing laser resource
        MettaGrid(from_mettagrid_config(game_config["game"]), game_map, 42)

    # Check the exception message
    assert "attack" in str(exc_info.value)
    assert "laser" in str(exc_info.value)
    assert "not in inventory_item_names" in str(exc_info.value)
    print(f"✓ Got expected exception: {exc_info.value}")


def test_no_exception_when_resources_in_inventory():
    """Test that no exception is raised when all consumed resources are in inventory."""
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", ".", "agent.blue", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    # Use create_test_config with all necessary resources in inventory
    game_config = create_test_config({
        "max_steps": 50,
        "num_agents": 2,
        "obs_width": 11,
        "obs_height": 11,
        "num_observation_tokens": 200,
        # Laser IS in inventory_item_names (already included by default in create_test_config)
        "inventory_item_names": ["laser", "armor", "heart"],
        "actions": {
            "attack": {"enabled": True, "consumed_resources": {"laser": 1}, "defense_resources": {"armor": 1}},
            "put_items": {"enabled": True},
            "get_items": {"enabled": True},
            "swap": {"enabled": True},
        },
        "groups": {"red": {"id": 0, "props": {}}, "blue": {"id": 1, "props": {}}},
        "agent": {"freeze_duration": 5},
    })

    # This should not raise an exception
    try:
        MettaGrid(from_mettagrid_config(game_config["game"]), game_map, 42)
        print("✓ No exception raised when all resources are in inventory")
    except ValueError as e:
        pytest.fail(f"Unexpected ValueError: {e}")


if __name__ == "__main__":
    test_exception_when_laser_not_in_inventory()
    test_no_exception_when_resources_in_inventory()
    print("\nAll tests passed!")
