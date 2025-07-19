#!/usr/bin/env python3
"""Test that converters can show recipe inputs in observations."""

import numpy as np
from metta.mettagrid.mettagrid_c import MettaGrid, ConverterConfig, PackedCoordinate
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
    inventory_items = ["ore", "fuel", "metal", "gold", "energy"]
    
    # Create converter configs with different recipes
    converter_with_recipe = ConverterConfig(
        type_id=10,
        type_name="smelter_with_recipe",
        input_resources={0: 2, 1: 1},  # 2 ore, 1 fuel
        output_resources={2: 1},        # 1 metal
        max_output=5,
        conversion_ticks=3,
        cooldown=2,
        initial_resource_count=0,
        color=100,
        show_recipe_inputs=True,
    )
    
    converter_no_recipe = ConverterConfig(
        type_id=11,
        type_name="smelter_no_recipe",
        input_resources={0: 2, 1: 1},  # Same recipe
        output_resources={2: 1},
        max_output=5,
        conversion_ticks=3,
        cooldown=2,
        initial_resource_count=0,
        color=101,
        show_recipe_inputs=False,
    )
    
    # Another converter with different recipe
    converter_complex_recipe = ConverterConfig(
        type_id=12,
        type_name="advanced_smelter",
        input_resources={2: 3, 4: 2},  # 3 metal, 2 energy
        output_resources={3: 1},        # 1 gold
        max_output=3,
        conversion_ticks=5,
        cooldown=3,
        initial_resource_count=0,
        color=102,
        show_recipe_inputs=True,
    )
    
    # Create game config
    game_config = {
        "num_agents": 1,
        "max_steps": 50,
        "obs_width": 9,
        "obs_height": 9,
        "num_observation_tokens": 200,
        "inventory_item_names": inventory_items,
        "groups": {"test": {"id": 0, "props": {}}},
        "agent": {
            "default_resource_limit": 10,
            "freeze_duration": 0,
            "rewards": {},
            "action_failure_penalty": 0.0,
        },
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "interact": {"enabled": True},
        },
        "objects": {
            "wall": {"type_id": 1},
            "agent.test": {"type_id": 0, "group": "test"},
            "smelter_with_recipe": converter_with_recipe,
            "smelter_no_recipe": converter_no_recipe,
            "advanced_smelter": converter_complex_recipe,
        },
    }
    
    # Create map with agent and converters
    game_map = [
        ["wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.test", "smelter_with_recipe", "advanced_smelter", "empty", "wall"],
        ["wall", "empty", "smelter_no_recipe", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall"],
    ]
    
    # Create environment
    env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)
    
    # Set up buffers
    observations = np.zeros((1, 200, 3), dtype=dtype_observations)
    terminals = np.zeros(1, dtype=dtype_terminals)
    truncations = np.zeros(1, dtype=dtype_truncations)
    rewards = np.zeros(1, dtype=dtype_rewards)
    env.set_buffers(observations, terminals, truncations, rewards)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Feature offset for inventory items
    INVENTORY_FEATURE_OFFSET = 14  # From ObservationFeature::ObservationFeatureCount
    
    # Helper function to collect all observation tokens for a location
    def get_all_tokens_at_location(obs, row, col):
        """Get all observation tokens at a specific grid location."""
        location = PackedCoordinate.pack(row, col)
        tokens = []
        for token in obs:
            if token[0] == location and token[0] != 0xFF:  # Not empty
                tokens.append((token[0], token[1], token[2]))
        return tokens
    
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
    
    # Helper to get all converter observations
    def get_converter_observations(obs):
        """Extract observations for all converters in view."""
        agent_obs = obs[0]  # First (and only) agent
        # Agent is at (1, 1), observation window is 9x9, so agent is at center (4, 4)
        # Converters are at:
        # - smelter_with_recipe: (1, 2) -> obs position (4, 5)
        # - advanced_smelter: (1, 3) -> obs position (4, 6)
        # - smelter_no_recipe: (2, 2) -> obs position (5, 5)
        
        return {
            "smelter_with_recipe": extract_features_at_location(agent_obs, 4, 5),
            "advanced_smelter": extract_features_at_location(agent_obs, 4, 6),
            "smelter_no_recipe": extract_features_at_location(agent_obs, 5, 5),
        }
    
    # Helper to get all converter tokens (raw observations)
    def get_converter_tokens(obs):
        """Extract raw observation tokens for all converters in view."""
        agent_obs = obs[0]  # First (and only) agent
        return {
            "smelter_with_recipe": get_all_tokens_at_location(agent_obs, 4, 5),
            "advanced_smelter": get_all_tokens_at_location(agent_obs, 4, 6),
            "smelter_no_recipe": get_all_tokens_at_location(agent_obs, 5, 5),
        }
    
    # Test 1: Verify exact observation tokens emitted
    print("Test 1: Verify exact observation tokens")
    converter_tokens = get_converter_tokens(obs)
    
    # Check smelter_with_recipe tokens
    tokens_with = converter_tokens["smelter_with_recipe"]
    print(f"\n🔍 smelter_with_recipe tokens: {len(tokens_with)} total")
    
    # Expected tokens for smelter_with_recipe
    expected_tokens_with = {
        0: 10,   # TypeId = 10
        5: 100,  # Color = 100
        6: 0,    # ConvertingOrCoolingDown = 0 (idle)
        INVENTORY_FEATURE_OFFSET + 0: 2,  # ore requirement = 2
        INVENTORY_FEATURE_OFFSET + 1: 1,  # fuel requirement = 1
    }
    
    # Verify each expected token exists
    for feature_id, expected_value in expected_tokens_with.items():
        found = False
        for _, f_id, value in tokens_with:
            if f_id == feature_id:
                assert value == expected_value, f"Feature {feature_id}: expected {expected_value}, got {value}"
                found = True
                break
        assert found, f"Missing expected feature {feature_id} in smelter_with_recipe"
    
    # Print all tokens for debugging
    for loc, feature, value in sorted(tokens_with, key=lambda x: x[1]):
        feature_name = f"Feature{feature}"
        if feature == 0:
            feature_name = "TypeId"
        elif feature == 5:
            feature_name = "Color"
        elif feature == 6:
            feature_name = "ConvertingOrCoolingDown"
        elif feature >= INVENTORY_FEATURE_OFFSET and feature < INVENTORY_FEATURE_OFFSET + len(inventory_items):
            item_idx = feature - INVENTORY_FEATURE_OFFSET
            feature_name = f"Recipe:{inventory_items[item_idx]}"
        print(f"   Token: location={loc}, {feature_name}={value}")
    
    # Check smelter_no_recipe tokens (should NOT have recipe tokens)
    tokens_no = converter_tokens["smelter_no_recipe"]
    print(f"\n🔍 smelter_no_recipe tokens: {len(tokens_no)} total")
    
    # Verify NO recipe tokens are present
    for _, feature_id, _ in tokens_no:
        assert feature_id < INVENTORY_FEATURE_OFFSET or feature_id >= INVENTORY_FEATURE_OFFSET + len(inventory_items), \
            f"Unexpected recipe feature {feature_id} in smelter_no_recipe"
    
    # Expected tokens for smelter_no_recipe (no recipe features)
    expected_tokens_no = {
        0: 11,   # TypeId = 11
        5: 101,  # Color = 101
        6: 0,    # ConvertingOrCoolingDown = 0 (idle)
    }
    
    for feature_id, expected_value in expected_tokens_no.items():
        found = False
        for _, f_id, value in tokens_no:
            if f_id == feature_id:
                assert value == expected_value, f"Feature {feature_id}: expected {expected_value}, got {value}"
                found = True
                break
        assert found, f"Missing expected feature {feature_id} in smelter_no_recipe"
    
    # Check advanced_smelter tokens
    tokens_adv = converter_tokens["advanced_smelter"]
    print(f"\n🔍 advanced_smelter tokens: {len(tokens_adv)} total")
    
    # Expected tokens for advanced_smelter
    expected_tokens_adv = {
        0: 12,   # TypeId = 12
        5: 102,  # Color = 102
        6: 0,    # ConvertingOrCoolingDown = 0 (idle)
        INVENTORY_FEATURE_OFFSET + 2: 3,  # metal requirement = 3
        INVENTORY_FEATURE_OFFSET + 4: 2,  # energy requirement = 2
    }
    
    for feature_id, expected_value in expected_tokens_adv.items():
        found = False
        for _, f_id, value in tokens_adv:
            if f_id == feature_id:
                assert value == expected_value, f"Feature {feature_id}: expected {expected_value}, got {value}"
                found = True
                break
        assert found, f"Missing expected feature {feature_id} in advanced_smelter"
    
    print("✅ Exact token verification passed")
    
    # Test 2: Verify observations during conversion
    print("\nTest 2: Verify observations during conversion")
    
    # Give agent resources
    env.add_resource(0, 0, 10)  # 10 ore
    env.add_resource(0, 1, 10)  # 10 fuel
    env.add_resource(0, 2, 10)  # 10 metal
    env.add_resource(0, 4, 10)  # 10 energy
    
    # Move agent to interact with smelter_with_recipe
    env.step([0])  # Move right to (1, 2)
    obs, _ = env.step([4])  # Interact (action 4)
    
    # Get tokens during conversion
    converter_tokens = get_converter_tokens(obs)
    tokens_converting = converter_tokens["smelter_with_recipe"]
    
    print(f"\n🔍 smelter_with_recipe tokens during conversion: {len(tokens_converting)} total")
    
    # Verify all expected tokens are still present during conversion
    expected_during_conversion = {
        0: 10,   # TypeId = 10
        5: 100,  # Color = 100
        6: 1,    # ConvertingOrCoolingDown = 1 (converting)
        INVENTORY_FEATURE_OFFSET + 0: 2,  # ore requirement = 2
        INVENTORY_FEATURE_OFFSET + 1: 1,  # fuel requirement = 1
    }
    
    for feature_id, expected_value in expected_during_conversion.items():
        found = False
        for _, f_id, value in tokens_converting:
            if f_id == feature_id:
                assert value == expected_value, f"Feature {feature_id}: expected {expected_value}, got {value}"
                found = True
                break
        assert found, f"Missing expected feature {feature_id} during conversion"
    
    # Verify exact token count (should be same as idle state)
    assert len(tokens_converting) == len(expected_during_conversion), \
        f"Expected {len(expected_during_conversion)} tokens, got {len(tokens_converting)}"
    
    print("✅ Conversion state observation verification passed")
    
    # Test 3: Verify observations during cooldown
    print("\nTest 3: Verify observations during cooldown")
    
    # Step through conversion
    for _ in range(3):  # conversion_ticks = 3
        obs, _ = env.step([0])  # Noop
    
    converter_tokens = get_converter_tokens(obs)
    tokens_cooldown = converter_tokens["smelter_with_recipe"]
    
    print(f"\n🔍 smelter_with_recipe tokens during cooldown: {len(tokens_cooldown)} total")
    
    # Verify all expected tokens are still present during cooldown
    expected_during_cooldown = {
        0: 10,   # TypeId = 10
        5: 100,  # Color = 100
        6: 1,    # ConvertingOrCoolingDown = 1 (cooling down)
        INVENTORY_FEATURE_OFFSET + 0: 2,  # ore requirement = 2
        INVENTORY_FEATURE_OFFSET + 1: 1,  # fuel requirement = 1
    }
    
    for feature_id, expected_value in expected_during_cooldown.items():
        found = False
        for _, f_id, value in tokens_cooldown:
            if f_id == feature_id:
                assert value == expected_value, f"Feature {feature_id}: expected {expected_value}, got {value}"
                found = True
                break
        assert found, f"Missing expected feature {feature_id} during cooldown"
    
    print("✅ Cooldown state observation verification passed")
    
    # Test 4: Comprehensive observation structure validation
    print("\nTest 4: Comprehensive observation structure validation")
    
    # Step through cooldown to return to idle
    for _ in range(2):  # cooldown = 2
        obs, _ = env.step([0])
    
    # Get feature spec for validation
    feature_spec = env.feature_spec()
    
    # Build expected observation structure
    print("\n📋 Complete observation structure validation:")
    
    converter_tokens = get_converter_tokens(obs)
    
    # Validate each converter's complete observation structure
    for converter_name, expected_config in [
        ("smelter_with_recipe", {
            "type_id": 10,
            "color": 100,
            "show_recipe": True,
            "recipe_inputs": {0: 2, 1: 1}  # ore: 2, fuel: 1
        }),
        ("smelter_no_recipe", {
            "type_id": 11,
            "color": 101,
            "show_recipe": False,
            "recipe_inputs": {}  # Should be empty
        }),
        ("advanced_smelter", {
            "type_id": 12,
            "color": 102,
            "show_recipe": True,
            "recipe_inputs": {2: 3, 4: 2}  # metal: 3, energy: 2
        })
    ]:
        tokens = converter_tokens[converter_name]
        print(f"\n   {converter_name}:")
        
        # Count tokens by category
        base_tokens = 0
        recipe_tokens = 0
        
        for _, feature_id, value in tokens:
            if feature_id < INVENTORY_FEATURE_OFFSET:
                base_tokens += 1
            else:
                recipe_tokens += 1
        
        print(f"     - Base feature tokens: {base_tokens}")
        print(f"     - Recipe feature tokens: {recipe_tokens}")
        
        # Verify token counts
        expected_recipe_tokens = len(expected_config["recipe_inputs"]) if expected_config["show_recipe"] else 0
        assert recipe_tokens == expected_recipe_tokens, \
            f"{converter_name}: expected {expected_recipe_tokens} recipe tokens, got {recipe_tokens}"
        
        # Verify each recipe input token
        if expected_config["show_recipe"]:
            for item_idx, expected_count in expected_config["recipe_inputs"].items():
                feature_id = INVENTORY_FEATURE_OFFSET + item_idx
                found = False
                for _, f_id, value in tokens:
                    if f_id == feature_id:
                        assert value == expected_count, \
                            f"{converter_name}: {inventory_items[item_idx]} expected {expected_count}, got {value}"
                        found = True
                        print(f"     - Recipe token: {inventory_items[item_idx]}={value} ✓")
                        break
                assert found, f"{converter_name}: missing recipe token for {inventory_items[item_idx]}"
    
    # Test 5: Verify no unexpected tokens
    print("\nTest 5: Verify no unexpected observation tokens")
    
    # Define all valid feature IDs
    valid_base_features = {0, 5, 6}  # TypeId, Color, ConvertingOrCoolingDown
    valid_inventory_features = set(range(INVENTORY_FEATURE_OFFSET, INVENTORY_FEATURE_OFFSET + len(inventory_items)))
    
    for converter_name, tokens in converter_tokens.items():
        for _, feature_id, _ in tokens:
            assert feature_id in valid_base_features or feature_id in valid_inventory_features, \
                f"{converter_name}: unexpected feature ID {feature_id}"
    
    print("✅ No unexpected tokens found")
    
    print("\n✅ All tests passed!")
    print("\n📊 Summary:")
    print("   - Converters emit exact expected observation tokens")
    print("   - Recipe tokens appear ONLY when show_recipe_inputs=True")
    print("   - Recipe tokens persist correctly during conversion and cooldown")
    print("   - No unexpected tokens are emitted")
    print("   - Token count and structure match expectations exactly")


if __name__ == "__main__":
    test_converter_recipe_observations()
