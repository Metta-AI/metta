"""Test that recipe inputs have different feature IDs than inventory items."""

# import pytest

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
    
    # Check that inventory features exist
    assert "inv:ore_red" in feature_spec
    assert "inv:battery_red" in feature_spec
    
    # Recipe inputs now use fixed feature IDs (recipe_input_0, recipe_input_1, etc.)
    # instead of offset-based IDs
    assert "recipe_input_0" in feature_spec
    
    # Get their IDs - they should be completely different
    inv_ore_id = feature_spec["inv:ore_red"]["id"]
    inv_battery_id = feature_spec["inv:battery_red"]["id"]
    recipe_0_id = feature_spec["recipe_input_0"]["id"]
    
    # Inventory items start at 24 (after all the fixed features including RecipeInput0-9)
    assert inv_ore_id == 24  # First inventory item
    assert inv_battery_id == 25  # Second inventory item
    
    # Recipe inputs use fixed IDs starting at 14
    assert recipe_0_id == 14  # RecipeInput0
    
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
    
    # The generator should show its recipe requirements via recipe_input_0
    # The encoded value contains both the item ID (0 for ore_red) and amount (2)
    # Encoded as: (item_id << 8) | amount = (0 << 8) | 2 = 2
    assert "recipe_input_0" in generator
    encoded_value = generator["recipe_input_0"]
    item_id = encoded_value >> 8
    amount = encoded_value & 0xFF
    assert item_id == 0  # ore_red is item 0
    assert amount == 2   # Recipe requires 2 ore_red
    
    # And it should show current inventory separately
    assert generator.get("inv:battery_red", 0) == 0  # No batteries in inventory yet


if __name__ == "__main__":
    test_recipe_input_feature_ids()
    print("\nTest passed! Recipe inputs now use dedicated feature IDs instead of inventory offsets.")