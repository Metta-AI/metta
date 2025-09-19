"""Test that converters show recipe details in observations correctly.

This test file verifies the behavior of converter observations with and without
the recipe_details_obs flag. When recipe_details_obs=True, converters show
their recipe inputs and outputs as separate features in their observations.
"""

from mettagrid.config.mettagrid_c_config import from_mettagrid_config
from mettagrid.mettagrid_c import MettaGrid, PackedCoordinate


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
            "resource_names": ["ore_red", "ore_blue", "battery_red", "heart"],
            "recipe_details_obs": False,  # Default to False
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "put_items": {"enabled": True},
                "get_items": {"enabled": True},
                "attack": {"enabled": False},
                "swap": {"enabled": False},
                "change_color": {"enabled": False},
                "change_glyph": {"enabled": False, "number_of_glyphs": 0},
            },
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

        input_output_feature_ids = {
            v["id"] for k, v in env.feature_spec().items() if k.startswith("input:") or k.startswith("output:")
        }

        # Extract feature IDs from tokens
        feature_ids = generator_tokens[:, 1]

        # No inventory features should be present (since converter has no actual inventory)
        inventory_features = [id for id in feature_ids if id in input_output_feature_ids]

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

        ore_red_input_id = env.feature_spec()["input:ore_red"]["id"]
        ore_blue_input_id = env.feature_spec()["input:ore_blue"]["id"]
        battery_red_output_id = env.feature_spec()["output:battery_red"]["id"]

        # Create a mapping of feature ID to value
        feature_map = {token[1]: token[2] for token in generator_tokens}

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

        ore_red_input_id = env.feature_spec()["input:ore_red"]["id"]
        ore_blue_input_id = env.feature_spec()["input:ore_blue"]["id"]
        battery_red_input_id = env.feature_spec()["input:battery_red"]["id"]

        assert ore_red_input_id in gen_feature_map and ore_blue_input_id in gen_feature_map, (
            f"Generator should have ore inputs at offsets {ore_red_input_id}, {ore_blue_input_id}"
        )
        assert battery_red_input_id in altar_feature_map, (
            f"Altar should have battery_red input at offset {battery_red_input_id}"
        )

        # Verify the recipe differences
        assert battery_red_input_id not in gen_feature_map, "Generator should not have battery_red input"
        assert ore_red_input_id not in altar_feature_map, "Altar should not have ore_red input"
        assert ore_blue_input_id not in altar_feature_map, "Altar should not have ore_blue input"
