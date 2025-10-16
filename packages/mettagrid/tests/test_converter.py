"""Tests covering converter observations and cooldown sequencing.

This file verifies converter observation tokens alongside cooldown schedule
behaviour. When recipe_details_obs=True, converters show their recipe inputs
and outputs as separate features. Additional tests confirm that converters
accept a list of cooldown values, cycle through them, and surface telemetry
through standard replay fields.
"""

import numpy as np

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
from mettagrid.mettagrid_c import MettaGrid, PackedCoordinate, dtype_actions


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
            cooldown=[10],
            initial_resource_count=0,
        )
        game_config.objects["altar"] = ConverterConfig(
            type_id=3,
            input_resources={"battery_red": 3},
            output_resources={"heart": 1},
            max_output=10,
            conversion_ticks=10,
            cooldown=[20],
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


def _build_converter_env(
    cooldown_schedule: list[int],
    max_conversions: int = -1,
) -> MettaGrid:
    """Create a small environment containing a single converter."""
    game_config = GameConfig(
        max_steps=200,
        num_agents=1,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=32,
        resource_names=["battery"],
        actions=ActionsConfig(
            noop=ActionConfig(enabled=True),
        ),
        agent=AgentConfig(
            default_resource_limit=10,
            freeze_duration=0,
            rewards=AgentRewards(),
        ),
        objects={
            "wall": WallConfig(type_id=1, swappable=False),
            "converter": ConverterConfig(
                type_id=2,
                input_resources={},
                output_resources={"battery": 1},
                max_output=-1,
                max_conversions=max_conversions,
                conversion_ticks=1,
                cooldown=cooldown_schedule,
                initial_resource_count=0,
            ),
        },
    )

    game_map = [
        ["wall", "wall", "wall"],
        ["wall", "agent.agent", "converter"],
        ["wall", "wall", "wall"],
    ]

    return MettaGrid(from_mettagrid_config(game_config), game_map, 99)


def _get_converter_object(env: MettaGrid) -> dict:
    """Fetch the converter entry from grid_objects output."""
    objects = env.grid_objects()
    for obj in objects.values():
        if obj.get("type_name") == "converter":
            return obj
    raise AssertionError("Converter not found in grid objects output")


def _cooldown_values_from_completions(
    completions: list[int],
    conversion_ticks: int,
) -> list[int]:
    """Derive cooldown durations used between consecutive completions."""
    used: list[int] = []
    for index in range(1, len(completions)):
        gap = completions[index] - completions[index - 1]
        cooldown = gap - conversion_ticks
        used.append(cooldown if cooldown > 0 else 0)
    return used


class TestConverterCooldownTime:
    """Test cooldown schedule runtime behaviour."""

    def test_runtime_sequence(self):
        """Converters cycle through provided cooldown schedules at runtime."""
        cases = [
            {
                "name": "single_value",
                "cooldown": [3],
                "expected_used": [3, 3, 3, 3],
            },
            {
                "name": "sequence",
                "cooldown": [2, 4, 0],
                "expected_used": [2, 4, 0, 2, 4],
            },
            {
                "name": "with_zero",
                "cooldown": [5, 0, 10],
                "expected_used": [5, 0, 10, 5],
            },
            {
                "name": "long_sequence",
                "cooldown": [1, 2, 3, 4, 5, 6],
                "expected_used": [1, 2, 3, 4, 5, 6],
            },
        ]

        for case in cases:
            schedule = case["cooldown"]
            env = _build_converter_env(schedule)
            env.reset()

            converter = _get_converter_object(env)

            noop_index = env.action_names().index("noop")
            actions = np.full(env.num_agents, noop_index, dtype=dtype_actions)

            completions: list[int] = []
            last_output = 0
            total_steps = 40

            for _ in range(total_steps):
                env.step(actions)
                converter = _get_converter_object(env)
                inventory = converter.get("inventory", {})
                output = int(inventory.get(0, 0))

                if output > last_output:
                    completions.append(env.current_step)
                    last_output = output

                expected_next = schedule[len(completions) % len(schedule)]
                observed_next = converter["cooldown_duration"]
                assert observed_next == expected_next, (
                    f"{case['name']} expected next cooldown {expected_next} but got {observed_next}"
                )

            used = _cooldown_values_from_completions(completions, conversion_ticks=1)
            expected_used = case["expected_used"]
            assert used[: len(expected_used)] == expected_used, case["name"]

    def test_max_conversions_respected(self):
        """Converters stop after reaching max_conversions limit."""
        env = _build_converter_env([5, 10], max_conversions=2)
        env.reset()

        noop_index = env.action_names().index("noop")
        actions = np.full(env.num_agents, noop_index, dtype=dtype_actions)

        completions: list[int] = []
        last_output = 0

        for _ in range(20):
            env.step(actions)
            converter = _get_converter_object(env)
            inventory = converter.get("inventory", {})
            output = int(inventory.get(0, 0))
            if output > last_output:
                completions.append(env.current_step)
                last_output = output

        assert len(completions) == 2
        assert last_output == 2

    def test_max_conversions_with_zero_cooldown(self):
        """Converters must not exceed max_conversions even with zero cooldown."""
        env = _build_converter_env([0, 0, 0], max_conversions=3)
        env.reset()

        noop_index = env.action_names().index("noop")
        actions = np.full(env.num_agents, noop_index, dtype=dtype_actions)

        completions: list[int] = []
        last_output = 0

        for _ in range(20):
            env.step(actions)
            converter = _get_converter_object(env)
            inventory = converter.get("inventory", {})
            output = int(inventory.get(0, 0))
            if output > last_output:
                completions.append(env.current_step)
                last_output = output

        assert len(completions) == 3, f"Expected 3 conversions but got {len(completions)}"
        assert last_output == 3, f"Expected output of 3 but got {last_output}"
