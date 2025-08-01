"""Test that converters show recipe details in observations correctly.

This test file verifies the behavior of converter observations with and without
the recipe_details_obs flag. When recipe_details_obs=True, converters show
their recipe inputs and outputs as separate features in their observations.
"""

from metta.mettagrid.mettagrid_c import MettaGrid, PackedCoordinate
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


class TestConverterObservations:
    """Test converter observations with and without recipe_details_obs."""

    def get_base_game_config(self):
        """Get base game configuration template."""
        return {
            "max_steps": 50,
            "num_agents": 1,
            "obs_width": 5,
            "obs_height": 5,
            "num_observation_tokens": 100,
            "inventory_item_names": ["ore_red", "ore_blue", "battery_red", "heart"],
            "recipe_details_obs": False,  # Default to False
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "put_items": {"enabled": True},
                "get_items": {"enabled": True},
                "attack": {"enabled": False},
                "swap": {"enabled": False},
                "change_color": {"enabled": False},
                "change_glyph": {"enabled": False, "number_of_glyphs": 0},
            },
            "groups": {"agent": {"id": 0, "props": {}}},
            "objects": {
                "wall": {"type_id": 1, "swappable": False},
            },
            "agent": {
                "default_resource_limit": 50,
                "freeze_duration": 0,
                "rewards": {"inventory": {"heart": 1.0}},
            },
        }

    def create_converter_env(self, recipe_details_obs=False):
        """Create a test environment with converters."""
        game_config = self.get_base_game_config()
        game_config["recipe_details_obs"] = recipe_details_obs

        # Add converter objects
        game_config["objects"]["generator"] = {
            "type_id": 2,
            "input_resources": {"ore_red": 2, "ore_blue": 1},
            "output_resources": {"battery_red": 1},
            "max_output": -1,
            "conversion_ticks": 5,
            "cooldown": 10,
            "initial_resource_count": 0,
            "color": 1,
        }
        game_config["objects"]["altar"] = {
            "type_id": 3,
            "input_resources": {"battery_red": 3},
            "output_resources": {"heart": 1},
            "max_output": 10,
            "conversion_ticks": 10,
            "cooldown": 20,
            "initial_resource_count": 0,
            "color": 2,
        }

        # Create a simple map with agent and converters
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", ".", "generator", "wall"],
            ["wall", ".", ".", ".", "wall"],
            ["wall", "altar", ".", ".", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        return MettaGrid(from_mettagrid_config(game_config), game_map, 42)

    def get_converter_tokens_at_location(self, obs, x, y):
        """Get all observation tokens for a converter at a specific location."""
        location = PackedCoordinate.pack(y, x)
        return obs[obs[:, 0] == location]

    def test_converter_obs_without_recipe_details(self):
        """Test that converters don't show recipe details when recipe_details_obs=False."""
        env = self.create_converter_env(recipe_details_obs=False)
        obs, _ = env.reset()
        agent_obs = obs[0]

        # Check generator at position (3, 1) - appears at (4, 2) in observation window
        generator_tokens = self.get_converter_tokens_at_location(agent_obs, 4, 2)

        # Extract feature IDs from tokens
        feature_ids = generator_tokens[:, 1]

        # Inventory features are IDs 14-17
        inventory_feature_start = 14
        inventory_feature_end = inventory_feature_start + 4  # 4 inventory items

        # No inventory features should be present (since converter has no actual inventory)
        inventory_features = feature_ids[
            (feature_ids >= inventory_feature_start) & (feature_ids < inventory_feature_end)
        ]

        assert len(inventory_features) == 0, (
            "No inventory/recipe features should be present when recipe_details_obs=False"
        )

    def test_converter_obs_with_recipe_details(self):
        """Test that converters show recipe details when recipe_details_obs=True."""
        env = self.create_converter_env(recipe_details_obs=True)
        obs, _ = env.reset()
        agent_obs = obs[0]

        # Check generator at position (3, 1) - appears at (4, 2) in observation window
        generator_tokens = self.get_converter_tokens_at_location(agent_obs, 4, 2)

        # Converters should emit TypeId, Color, ConvertingOrCoolingDown, and recipe inputs
        assert len(generator_tokens) >= 3, f"Converter should emit at least 3 tokens, got {len(generator_tokens)}"

        # Extract feature IDs and values from tokens
        feature_ids = generator_tokens[:, 1]
        feature_values = generator_tokens[:, 2]

        # Create a mapping of feature ID to value
        feature_map = {fid: val for fid, val in zip(feature_ids, feature_values, strict=False)}

        # Calculate dynamic offsets based on inventory item count
        inventory_item_count = 4  # ore_red, ore_blue, battery_red, heart
        input_recipe_offset = 15 + inventory_item_count  # 19 (was 18, but ObservationFeatureCount is now 15)
        output_recipe_offset = input_recipe_offset + inventory_item_count  # 23 (was 22)

        # Recipe inputs are shown at dynamic offsets
        # ore_red=input_recipe_offset+0, ore_blue=input_recipe_offset+1, etc.
        ore_red_input_id = input_recipe_offset + 0  # 18
        ore_blue_input_id = input_recipe_offset + 1  # 19
        battery_red_output_id = output_recipe_offset + 2  # 24

        assert ore_red_input_id in feature_map, f"Should have ore_red recipe input at offset {ore_red_input_id}"
        assert ore_blue_input_id in feature_map, f"Should have ore_blue recipe input at offset {ore_blue_input_id}"

        # Generator expects: ore_red=2, ore_blue=1
        assert feature_map[ore_red_input_id] == 2, f"ore_red input should be 2, got {feature_map[ore_red_input_id]}"
        assert feature_map[ore_blue_input_id] == 1, f"ore_blue input should be 1, got {feature_map[ore_blue_input_id]}"

        # Recipe outputs are shown at dynamic offsets
        assert battery_red_output_id in feature_map, (
            f"Should have battery_red recipe output at offset {battery_red_output_id}"
        )
        assert feature_map[battery_red_output_id] == 1, (
            f"battery_red output should be 1, got {feature_map[battery_red_output_id]}"
        )

    def test_altar_obs_with_recipe_details(self):
        """Test altar observations with recipe details enabled."""
        env = self.create_converter_env(recipe_details_obs=True)
        obs, _ = env.reset()
        agent_obs = obs[0]

        # Check altar at position (1, 3) - appears at (2, 4) in observation window
        altar_tokens = self.get_converter_tokens_at_location(agent_obs, 2, 4)

        # Altar should emit at least TypeId, Color, and ConvertingOrCoolingDown
        assert len(altar_tokens) >= 3, f"Altar should emit at least 3 tokens, got {len(altar_tokens)}"

        # Extract feature IDs and values
        feature_ids = altar_tokens[:, 1]
        feature_values = altar_tokens[:, 2]

        # Create a mapping of feature ID to value
        feature_map = {fid: val for fid, val in zip(feature_ids, feature_values, strict=False)}

        # Calculate dynamic offsets based on inventory item count
        inventory_item_count = 4  # ore_red, ore_blue, battery_red, heart
        input_recipe_offset = 15 + inventory_item_count  # 19 (was 18, but ObservationFeatureCount is now 15)
        output_recipe_offset = input_recipe_offset + inventory_item_count  # 23 (was 22)

        # Altar expects: battery_red=3 (feature ID is input_recipe_offset + 2)
        battery_red_input_id = input_recipe_offset + 2  # 21 (was 20)
        heart_output_id = output_recipe_offset + 3  # 26 (was 25)

        assert battery_red_input_id in feature_map, (
            f"Should have battery_red recipe input at offset {battery_red_input_id}"
        )
        assert feature_map[battery_red_input_id] == 3, (
            f"battery_red input should be 3, got {feature_map[battery_red_input_id]}"
        )

        # Altar produces: heart=1
        assert heart_output_id in feature_map, f"Should have heart recipe output at offset {heart_output_id}"
        assert feature_map[heart_output_id] == 1, f"heart output should be 1, got {feature_map[heart_output_id]}"

    def test_multiple_converters_different_recipes(self):
        """Test that different converters show their own recipe details correctly."""
        env = self.create_converter_env(recipe_details_obs=True)
        obs, _ = env.reset()
        agent_obs = obs[0]

        # Get tokens for both converters
        generator_tokens = self.get_converter_tokens_at_location(agent_obs, 4, 2)
        altar_tokens = self.get_converter_tokens_at_location(agent_obs, 2, 4)

        # Both should emit at least 3 tokens
        assert len(generator_tokens) >= 3, f"Generator should emit at least 3 tokens, got {len(generator_tokens)}"
        assert len(altar_tokens) >= 3, f"Altar should emit at least 3 tokens, got {len(altar_tokens)}"

        # Extract recipe features for generator
        gen_feature_map = {token[1]: token[2] for token in generator_tokens}

        # Extract recipe features for altar
        altar_feature_map = {token[1]: token[2] for token in altar_tokens}

        # Calculate dynamic offsets
        inventory_item_count = 4  # ore_red, ore_blue, battery_red, heart
        input_recipe_offset = 15 + inventory_item_count  # 19 (was 18, but ObservationFeatureCount is now 15)

        # Verify they have different recipe requirements
        # Generator has ore_red and ore_blue inputs
        ore_red_input_id = input_recipe_offset + 0  # 18
        ore_blue_input_id = input_recipe_offset + 1  # 19
        battery_red_input_id = input_recipe_offset + 2  # 20

        assert ore_red_input_id in gen_feature_map and ore_blue_input_id in gen_feature_map, (
            f"Generator should have ore inputs at offsets {ore_red_input_id}, {ore_blue_input_id}"
        )
        assert battery_red_input_id in altar_feature_map, (
            f"Altar should have battery_red input at offset {battery_red_input_id}"
        )

        # Verify the recipe differences
        assert battery_red_input_id not in gen_feature_map, "Generator should not have battery_red input"
        assert ore_red_input_id not in altar_feature_map and ore_blue_input_id not in altar_feature_map, (
            "Altar should not have ore inputs"
        )

    def test_converter_with_no_inputs(self):
        """Test converter that produces output without requiring inputs."""
        game_config = self.get_base_game_config()
        game_config["recipe_details_obs"] = True
        game_config["inventory_item_names"] = ["ore_red"]  # Only one inventory item

        # Add mine object
        game_config["objects"]["mine"] = {
            "type_id": 2,
            "input_resources": {},  # No inputs required
            "output_resources": {"ore_red": 1},
            "max_output": -1,
            "conversion_ticks": 1,
            "cooldown": 0,
            "initial_resource_count": 0,
            "color": 1,
        }

        game_map = [
            ["wall", "wall", "wall"],
            ["wall", "agent.agent", "mine"],
            ["wall", "wall", "wall"],
        ]

        env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)
        obs, _ = env.reset()
        agent_obs = obs[0]

        # Check mine at position (2, 1) - appears at (3, 2) in observation window
        mine_tokens = self.get_converter_tokens_at_location(agent_obs, 3, 2)
        feature_ids = mine_tokens[:, 1]

        # Mine has no input requirements, so should only have base features
        inventory_features = feature_ids[(feature_ids >= 14) & (feature_ids < 15)]
        assert len(inventory_features) == 0, "Mine with no inputs should have no recipe features"
