#!/usr/bin/env python3
"""Test that converters can show recipe inputs in observations."""

import numpy as np
from metta.mettagrid.mettagrid_c import MettaGrid, PackedCoordinate
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config
from metta.mettagrid.mettagrid_env import (
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)


def test_converter_recipe_observations():
    """Test that converters emit recipe inputs in observations when configured."""
    
    # Define inventory items
    inventory_items = ["ore", "fuel", "metal"]
    
    # Create game config
    game_config = {
        "num_agents": 1,
        "max_steps": 10,
        "obs_width": 7,
        "obs_height": 7,
        "num_observation_tokens": 100,
        "inventory_item_names": inventory_items,
        "groups": {"test": {"id": 0, "props": {}}},
        "agent": {
            "default_resource_limit": 10,
            "freeze_duration": 0,
            "rewards": {},
            "action_failure_penalty": 0.0,
        },
        "actions": {"noop": {"enabled": True}},
        "objects": {
            "wall": {"type_id": 1},
            "smelter_with_recipe": {
                "type_id": 10,
                "type_name": "smelter_with_recipe",
                "input_resources": {"0": 2, "1": 1},  # 2 ore, 1 fuel - string keys for JSON
                "output_resources": {"2": 1},         # 1 metal
                "max_output": 5,
                "conversion_ticks": 10,
                "cooldown": 5,
                "initial_resource_count": 0,
                "color": 100,
                "show_recipe_inputs": True,
            },
            "smelter_no_recipe": {
                "type_id": 11,
                "type_name": "smelter_no_recipe",
                "input_resources": {"0": 2, "1": 1},  # Same recipe - string keys for JSON
                "output_resources": {"2": 1},
                "max_output": 5,
                "conversion_ticks": 10,
                "cooldown": 5,
                "initial_resource_count": 0,
                "color": 101,
                "show_recipe_inputs": False,
            },
        },
    }
    
    # Create map with agent and converters
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.test", "smelter_with_recipe", "empty", "wall"],
        ["wall", "empty", "smelter_no_recipe", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]
    
    # Create environment
    env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)
    
    # Set up buffers
    observations = np.zeros((1, 100, 3), dtype=dtype_observations)
    terminals = np.zeros(1, dtype=dtype_terminals)
    truncations = np.zeros(1, dtype=dtype_truncations)
    rewards = np.zeros(1, dtype=dtype_rewards)
    env.set_buffers(observations, terminals, truncations, rewards)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Get grid objects to analyze converter observations
    grid_objects = env.grid_objects()
    
    # Find converters in grid objects
    converter_with_recipe_obj = None
    converter_no_recipe_obj = None
    
    for obj_id, obj_data in grid_objects.items():
        if obj_data.get("type_name") == "smelter_with_recipe":
            converter_with_recipe_obj = obj_data
        elif obj_data.get("type_name") == "smelter_no_recipe":
            converter_no_recipe_obj = obj_data
    
    assert converter_with_recipe_obj is not None, "Converter with recipe not found"
    assert converter_no_recipe_obj is not None, "Converter without recipe not found"
    
    # Analyze observations to find converter tokens
    # Feature offset for inventory items
    INVENTORY_FEATURE_OFFSET = 14  # From ObservationFeature::ObservationFeatureCount
    
    # Helper function to extract features from observation
    def extract_features_at_location(obs, row, col):
        """Extract all features at a specific grid location."""
        location = PackedCoordinate.pack(row, col)
        features = {}
        for token in obs:
            if token[0] == location and token[0] != 0xFF:  # Not empty
                feature_id = token[1]
                value = token[2]
                features[feature_id] = value
        return features
    
    # Agent is at (1, 1), converters are at (1, 2) and (2, 2)
    # In observation window (7x7), agent is at center (3, 3)
    # So converter_with_recipe is at observation position (3, 4)
    # And converter_no_recipe is at observation position (4, 4)
    
    agent_obs = obs[0]  # First (and only) agent
    
    # Extract features for converter with recipe (relative position: +1 col, 0 row)
    features_with_recipe = extract_features_at_location(agent_obs, 3, 4)
    
    # Extract features for converter without recipe (relative position: +1 row, +1 col)
    features_no_recipe = extract_features_at_location(agent_obs, 4, 4)
    
    # Verify both converters have basic features
    assert 0 in features_with_recipe, "Converter with recipe missing TypeId"
    assert 5 in features_with_recipe, "Converter with recipe missing Color"
    assert 6 in features_with_recipe, "Converter with recipe missing ConvertingOrCoolingDown"
    
    assert 0 in features_no_recipe, "Converter without recipe missing TypeId"
    assert 5 in features_no_recipe, "Converter without recipe missing Color"
    assert 6 in features_no_recipe, "Converter without recipe missing ConvertingOrCoolingDown"
    
    # Check that converters have correct type IDs
    assert features_with_recipe[0] == 10, f"Wrong type ID for converter with recipe: {features_with_recipe[0]}"
    assert features_no_recipe[0] == 11, f"Wrong type ID for converter without recipe: {features_no_recipe[0]}"
    
    # Check for recipe inputs in observations
    # The converter with show_recipe_inputs=True should include:
    # - ore (item 0): 2 units
    # - fuel (item 1): 1 unit
    ore_feature_id = INVENTORY_FEATURE_OFFSET + 0
    fuel_feature_id = INVENTORY_FEATURE_OFFSET + 1
    
    # Converter WITH recipe should show the recipe inputs
    assert ore_feature_id in features_with_recipe, "Converter with recipe should show ore requirement"
    assert features_with_recipe[ore_feature_id] == 2, f"Expected 2 ore, got {features_with_recipe[ore_feature_id]}"
    
    assert fuel_feature_id in features_with_recipe, "Converter with recipe should show fuel requirement"
    assert features_with_recipe[fuel_feature_id] == 1, f"Expected 1 fuel, got {features_with_recipe[fuel_feature_id]}"
    
    # Converter WITHOUT recipe should NOT show the recipe inputs
    assert ore_feature_id not in features_no_recipe, "Converter without recipe should not show ore requirement"
    assert fuel_feature_id not in features_no_recipe, "Converter without recipe should not show fuel requirement"
    
    print("✅ Converter recipe observation test passed!")
    print(f"   - Converter with recipe shows inputs: ore={features_with_recipe[ore_feature_id]}, fuel={features_with_recipe[fuel_feature_id]}")
    print(f"   - Converter without recipe hides inputs (as expected)")
    
    # Additional verification: check feature names
    feature_spec = env.feature_spec()
    print(f"\n📋 Feature mappings:")
    print(f"   - Ore (item 0) -> feature {ore_feature_id}: {feature_spec.get(f'inv:ore', {}).get('id', 'not found')}")
    print(f"   - Fuel (item 1) -> feature {fuel_feature_id}: {feature_spec.get(f'inv:fuel', {}).get('id', 'not found')}")


if __name__ == "__main__":
    test_converter_recipe_observations()
