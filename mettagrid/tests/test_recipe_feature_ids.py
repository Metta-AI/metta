"""Test that recipe inputs have different feature IDs than inventory items."""

import pytest

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


def test_recipe_input_feature_ids():
    """Test that converters with show_recipe_inputs use different feature IDs for recipe vs inventory."""
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent", ".", "generator", "wall"],
        ["wall", ".", ".", ".", "wall"],
        ["wall", "mine", ".", ".", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]
    
    game_config = {
        "max_steps": 100,
        "num_agents": 1,
        "obs_width": 5,
        "obs_height": 5,
        "num_observation_tokens": 100,
        "inventory_item_names": ["ore_red", "battery_red"],
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "get_items": {"enabled": True},
            "put_items": {"enabled": True},
        },
        "groups": {"agent": {"id": 0, "props": {}}},
        "objects": {
            "wall": {"type_id": 1},
            "mine": {
                "type_id": 2,
                "output_resources": {"ore_red": 3},
                "initial_resource_count": 3,
                "max_output": 10,
                "conversion_ticks": 1,
                "cooldown": 5,
                "show_recipe_inputs": False,  # Mines don't need to show recipe inputs
            },
            "generator": {
                "type_id": 3,
                "input_resources": {"ore_red": 2},  # Needs 2 ore_red
                "output_resources": {"battery_red": 1},
                "initial_resource_count": 0,
                "max_output": 10,
                "conversion_ticks": 1,
                "cooldown": 5,
                "show_recipe_inputs": True,  # Show recipe inputs
            },
        },
        "agent": {"default_resource_limit": 10, "rewards": {}},
    }
    
    env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)
    
    # Get feature specs to see the IDs
    feature_spec = env.feature_spec()
    
    # Check that inventory and recipe features have different IDs
    # ore_red inventory feature should be "inv:ore_red"
    # ore_red recipe feature should be "recipe:ore_red"
    assert "inv:ore_red" in feature_spec
    assert "recipe:ore_red" in feature_spec
    
    # Get their IDs
    inv_ore_id = feature_spec["inv:ore_red"]["id"]
    recipe_ore_id = feature_spec["recipe:ore_red"]["id"]
    
    # They should be different
    assert inv_ore_id != recipe_ore_id
    
    # Based on our constants, inventory items start at 14 (InventoryFeatureOffset)
    # and recipe items start at 50 (RecipeInputFeatureOffset)
    assert inv_ore_id == 14  # First inventory item
    assert recipe_ore_id == 50  # First recipe item
    
    # Also check battery_red
    assert feature_spec["inv:battery_red"]["id"] == 15  # Second inventory item
    assert feature_spec["recipe:battery_red"]["id"] == 51  # Second recipe item
    
    # Now let's check the actual observations
    obs, _, _, _ = env.reset()
    
    # Get grid objects to examine the generator's observation features
    grid_objects = env.grid_objects()
    
    # Find the generator object
    generator = None
    for obj_id, obj in grid_objects.items():
        if obj["type"] == 3:  # Generator type_id
            generator = obj
            break
    
    assert generator is not None
    print(f"\nGenerator features:")
    for key, value in generator.items():
        if key not in ["id", "type", "type_name", "r", "c", "layer"]:
            print(f"  {key}: {value}")
    
    # The generator should show recipe:ore_red = 2 (the input requirement)
    # and inv:battery_red = 0 (current inventory)
    assert generator.get("recipe:ore_red") == 2  # Recipe requires 2 ore_red
    assert generator.get("inv:battery_red") == 0  # No batteries in inventory yet


if __name__ == "__main__":
    test_recipe_input_feature_ids()
    print("\nTest passed! Recipe inputs now use different feature IDs than inventory items.")