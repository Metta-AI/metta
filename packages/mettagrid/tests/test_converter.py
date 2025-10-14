"""Test that converters show recipe details in observations correctly.

This test file verifies the behavior of converter observations with and without
the recipe_details_obs flag. When recipe_details_obs=True, converters show
their recipe inputs and outputs as separate features in their observations.
"""

import math
import struct

from mettagrid.config.mettagrid_c_config import from_mettagrid_config
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AttackActionConfig,
    ChangeGlyphActionConfig,
    ConverterConfig,
    GameConfig,
    WallConfig,
)
from mettagrid.mettagrid_c import MettaGrid, PackedCoordinate


class TestConverterObservations:
    """Test converter observations with and without recipe_details_obs."""

    def get_base_game_config(self, recipe_details_obs: bool = False) -> GameConfig:
        """Get base game configuration template."""
        return GameConfig(
            max_steps=50,
            num_agents=1,
            obs_width=5,
            obs_height=5,
            num_observation_tokens=100,
            resource_names=["ore_red", "ore_blue", "battery_red", "heart"],
            recipe_details_obs=recipe_details_obs,
            actions=ActionsConfig(
                noop=ActionConfig(enabled=True),
                move=ActionConfig(enabled=True),
                put_items=ActionConfig(enabled=True),
                get_items=ActionConfig(enabled=True),
                attack=AttackActionConfig(enabled=False),
                swap=ActionConfig(enabled=False),
                change_glyph=ChangeGlyphActionConfig(enabled=False, number_of_glyphs=0),
            ),
            objects={
                "wall": WallConfig(type_id=1, swappable=False),
            },
            agent=AgentConfig(
                default_resource_limit=50,
                freeze_duration=0,
                rewards=AgentRewards(inventory={"heart": 1.0}),
            ),
        )

    def create_converter_env(self, recipe_details_obs=False):
        """Create a test environment with converters."""
        game_config = self.get_base_game_config(recipe_details_obs=recipe_details_obs)

        # Add converter objects
        game_config.objects["generator"] = ConverterConfig(
            type_id=2,
            input_resources={"ore_red": 2, "ore_blue": 1},
            output_resources={"battery_red": 1},
            max_output=-1,
            conversion_ticks=5,
            cooldown=10,
            initial_resource_count=0,
        )
        game_config.objects["altar"] = ConverterConfig(
            type_id=3,
            input_resources={"battery_red": 3},
            output_resources={"heart": 1},
            max_output=10,
            conversion_ticks=10,
            cooldown=20,
            initial_resource_count=0,
        )

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

    def test_converter_fractional_outputs_emitted(self):
        """Test that fractional outputs are emitted as output_frac:* features with uint8 bucketization."""
        game_config = self.get_base_game_config(recipe_details_obs=True)
        # Add a converter with purely fractional output
        game_config.objects["frac_gen"] = ConverterConfig(
            type_id=4,
            input_resources={},
            output_resources={"battery_red": 0.5},
            max_output=-1,
            conversion_ticks=0,
            cooldown=0,
            initial_resource_count=0,
        )

        # Map with the fractional converter in view
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", ".", "frac_gen", "wall"],
            ["wall", ".", ".", ".", "wall"],
            ["wall", ".", ".", ".", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        env = MettaGrid(from_mettagrid_config(game_config), game_map, 7)
        obs, _ = env.reset()
        agent_obs = obs[0]

        # Converter at (3,1) shows up at (4,2)
        generator_tokens = self.get_converter_tokens_at_location(agent_obs, 4, 2)

        out_frac_id = env.feature_spec()["output_frac:battery_red"]["id"]
        out_id = env.feature_spec()["output:battery_red"]["id"]

        feature_map = {token[1]: token[2] for token in generator_tokens}

        assert out_frac_id in feature_map, "Should emit output_frac for fractional-only output"
        # 0.5 => ceil(0.5*255) = 128
        assert feature_map[out_frac_id] == 128, f"Expected 128 bucket for 0.5, got {feature_map[out_frac_id]}"
        assert out_id not in feature_map, "Integral output feature should be absent for <1.0 amount"

    def test_converter_fractional_remainder_emitted_for_amounts_above_one(self):
        """Test that outputs >=1.0 emit an output_frac:* token for their fractional remainder."""
        game_config = self.get_base_game_config(recipe_details_obs=True)
        game_config.objects["hybrid_gen"] = ConverterConfig(
            type_id=5,
            input_resources={},
            output_resources={"battery_red": 1.2},
            max_output=-1,
            conversion_ticks=0,
            cooldown=0,
            initial_resource_count=0,
        )

        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", ".", "hybrid_gen", "wall"],
            ["wall", ".", ".", ".", "wall"],
            ["wall", ".", ".", ".", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        env = MettaGrid(from_mettagrid_config(game_config), game_map, 11)
        obs, _ = env.reset()
        agent_obs = obs[0]

        converter_tokens = self.get_converter_tokens_at_location(agent_obs, 4, 2)

        out_frac_id = env.feature_spec()["output_frac:battery_red"]["id"]
        out_id = env.feature_spec()["output:battery_red"]["id"]

        feature_map = {token[1]: token[2] for token in converter_tokens}

        assert out_id in feature_map, "Should emit integral output component for amounts >= 1.0"
        assert feature_map[out_id] == 1, f"Integral component should be floored to 1, got {feature_map[out_id]}"

        assert out_frac_id in feature_map, "Should emit fractional remainder for amounts >= 1.0"
        float_amount = struct.unpack("!f", struct.pack("!f", 1.2))[0]
        remainder = float_amount - math.floor(float_amount)
        expected_bucket = math.ceil(remainder * 255.0)
        assert feature_map[out_frac_id] == expected_bucket, (
            f"Expected bucket {expected_bucket} for remainder {remainder}, got {feature_map[out_frac_id]}"
        )
