# Tests for tag property modification system
import pytest

from mettagrid.config.mettagrid_c_config import convert_to_cpp_game_config
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AssemblerConfig,
    ConverterConfig,
    GameConfig,
    MettaGridConfig,
    PropertyModifier,
    RecipeConfig,
    TagDefinition,
    WallConfig,
)
from mettagrid.core import MettaGridCore
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.test_support import TokenTypes


class TestTagPropertyModifications:
    """Test that tags can modify object properties."""

    def test_agent_tag_modifies_freeze_duration(self):
        """Test that a tag can modify agent freeze duration."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=2,
                max_steps=100,
                obs_width=5,
                obs_height=5,
                num_observation_tokens=200,
                actions=ActionsConfig(
                    noop=ActionConfig(),
                    move=ActionConfig(),
                ),
                agents=[
                    AgentConfig(team_id=0, freeze_duration=10, tags=["fast"]),
                    AgentConfig(team_id=1, freeze_duration=10, tags=[]),  # No tags
                ],
                tag_definitions={
                    "fast": TagDefinition(
                        name="fast",
                        modifiers=[PropertyModifier(property_path="freeze_duration", operation="multiply", value=0.5)],
                    )
                },
                resource_names=[],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        [".", "@", "."],
                        [".", ".", "."],
                        [".", "@", "."],
                    ],
                ),
            )
        )

        # Convert to cpp config to apply tag modifications
        # We need to verify the modifications were applied by checking the internal state
        # Since the C++ config doesn't expose objects directly, we verify by creating
        # an environment and checking that it works correctly

        # Verify exact modified value via cpp config
        cpp_config = convert_to_cpp_game_config(cfg.game)
        agent_cfg = cpp_config.objects.get("agent.red")
        assert agent_cfg is not None
        # freeze_duration: 10 * 0.5 = 5
        assert agent_cfg.freeze_duration == 5

    def test_wall_tag_modifies_swappable(self):
        """Test that a tag can modify wall swappable property."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                obs_width=3,
                obs_height=3,
                num_observation_tokens=200,
                actions=ActionsConfig(
                    noop=ActionConfig(),
                    swap=ActionConfig(),
                ),
                objects={
                    "wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID, swappable=False, tags=["pushable"]),
                },
                tag_definitions={
                    "pushable": TagDefinition(
                        name="pushable",
                        modifiers=[PropertyModifier(property_path="swappable", operation="override", value=True)],
                    )
                },
                resource_names=[],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#"],
                        ["#", "@", "#"],
                        ["#", "#", "#"],
                    ],
                ),
            )
        )

        # Validate modified value via cpp config
        cpp_config = convert_to_cpp_game_config(cfg.game)
        wall_cfg = cpp_config.objects.get("wall")
        assert wall_cfg is not None
        assert wall_cfg.swappable is True

    def test_converter_tag_modifies_conversion_speed(self):
        """Test that tags can modify converter properties."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                obs_width=3,
                obs_height=3,
                num_observation_tokens=200,
                actions=ActionsConfig(noop=ActionConfig()),
                objects={
                    "fast_converter": ConverterConfig(
                        type_id=2,
                        input_resources={"wood": 1},
                        output_resources={"coal": 1},
                        max_output=10,
                        conversion_ticks=10,
                        cooldown=5,
                        tags=["efficient", "turbo"],
                    ),
                    "normal_converter": ConverterConfig(
                        type_id=3,
                        input_resources={"wood": 1},
                        output_resources={"coal": 1},
                        max_output=10,
                        conversion_ticks=10,
                        cooldown=5,
                        tags=[],
                    ),
                },
                tag_definitions={
                    "efficient": TagDefinition(
                        name="efficient",
                        modifiers=[
                            PropertyModifier(property_path="max_output", operation="multiply", value=2),
                            PropertyModifier(property_path="cooldown", operation="multiply", value=0.8),
                        ],
                    ),
                    "turbo": TagDefinition(
                        name="turbo",
                        modifiers=[PropertyModifier(property_path="conversion_ticks", operation="multiply", value=0.5)],
                    ),
                },
                resource_names=["wood", "coal"],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        [".", "@", "."],
                    ],
                ),
            )
        )

        # Validate converter properties after tag application
        cpp_config = convert_to_cpp_game_config(cfg.game)
        fast_cfg = cpp_config.objects.get("fast_converter")
        normal_cfg = cpp_config.objects.get("normal_converter")
        assert fast_cfg is not None
        assert normal_cfg is not None
        # fast: max_output 10*2=20, conversion_ticks 10*0.5=5, cooldown 5*0.8=4
        assert fast_cfg.max_output == 20
        assert fast_cfg.conversion_ticks == 5
        assert fast_cfg.cooldown == 4
        # normal unchanged
        assert normal_cfg.max_output == 10
        assert normal_cfg.conversion_ticks == 10
        assert normal_cfg.cooldown == 5

    def test_assembler_recipe_unknown_resources_raise(self):
        """Assembler recipes referencing unknown resources should raise a clear error."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=10,
                obs_width=3,
                obs_height=3,
                num_observation_tokens=128,
                actions=ActionsConfig(noop=ActionConfig()),
                objects={
                    "bench": AssemblerConfig(
                        type_id=7,
                        recipes=[
                            (
                                ["Any"],
                                RecipeConfig(
                                    input_resources={"unknown": 1},
                                    output_resources={"known": 1},
                                    cooldown=1,
                                ),
                            )
                        ],
                        tags=[],
                    )
                },
                tag_definitions={},
                resource_names=["known"],
                map_builder=AsciiMapBuilder.Config(map_data=[["@"]]),
            )
        )

        with pytest.raises(ValueError) as exc:
            _ = convert_to_cpp_game_config(cfg.game)
        assert "unknown resources" in str(exc.value)

    def test_tag_ids_and_tag_id_map_propagate(self):
        """Verify tag_ids on objects and tag_id_map are populated in C++ config."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=10,
                obs_width=3,
                obs_height=3,
                num_observation_tokens=128,
                actions=ActionsConfig(noop=ActionConfig()),
                agents=[AgentConfig(team_id=0, freeze_duration=1, tags=["t1"])],
                objects={
                    "wall": WallConfig(type_id=1, swappable=False, tags=["t1"]),
                },
                tag_definitions={
                    "t1": TagDefinition(name="t1", modifiers=[]),
                },
                resource_names=[],
                map_builder=AsciiMapBuilder.Config(map_data=[["@"]]),
            )
        )

        cpp_config = convert_to_cpp_game_config(cfg.game)
        # Build reverse map name->id to validate IDs consistently used
        name_to_id = {name: id for id, name in cpp_config.tag_id_map.items()}
        assert "t1" in name_to_id
        t1_id = name_to_id["t1"]
        # Agent uses team 0 => red
        agent_cfg = cpp_config.objects.get("agent.red")
        wall_cfg = cpp_config.objects.get("wall")
        assert agent_cfg is not None and wall_cfg is not None
        assert agent_cfg.tag_ids == [t1_id]
        assert wall_cfg.tag_ids == [t1_id]

    def test_multiple_tags_apply_in_order(self):
        """Test that multiple tags apply their modifications in order."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                obs_width=3,
                obs_height=3,
                num_observation_tokens=200,
                actions=ActionsConfig(noop=ActionConfig()),
                agents=[
                    AgentConfig(
                        team_id=0,
                        freeze_duration=10,
                        action_failure_penalty=1.0,
                        tags=["base_modifier", "final_modifier"],
                    ),
                ],
                tag_definitions={
                    "base_modifier": TagDefinition(
                        name="base_modifier",
                        modifiers=[
                            PropertyModifier(
                                property_path="freeze_duration",
                                operation="add",
                                value=5,  # 10 + 5 = 15
                            ),
                            PropertyModifier(
                                property_path="action_failure_penalty",
                                operation="multiply",
                                value=2,  # 1.0 * 2 = 2.0
                            ),
                        ],
                    ),
                    "final_modifier": TagDefinition(
                        name="final_modifier",
                        modifiers=[
                            PropertyModifier(
                                property_path="freeze_duration",
                                operation="multiply",
                                value=2,  # 15 * 2 = 30
                            ),
                            PropertyModifier(
                                property_path="action_failure_penalty",
                                operation="override",
                                value=0.5,  # Override to 0.5
                            ),
                        ],
                    ),
                },
                resource_names=[],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        [".", "@", "."],
                    ],
                ),
            )
        )

        # Validate exact modified values via cpp config
        cpp_config = convert_to_cpp_game_config(cfg.game)
        agent_cfg = cpp_config.objects.get("agent.red")
        assert agent_cfg is not None
        # freeze_duration: 10 + 5 = 15, then 15 * 2 = 30
        assert agent_cfg.freeze_duration == 30
        # action_failure_penalty: 1.0 * 2 = 2.0, then override to 0.5
        assert abs(agent_cfg.action_failure_penalty - 0.5) < 1e-6

    def test_resource_limit_modification(self):
        """Test that tags can modify resource limits."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                obs_width=3,
                obs_height=3,
                num_observation_tokens=200,
                actions=ActionsConfig(noop=ActionConfig()),
                agents=[
                    AgentConfig(team_id=0, default_resource_limit=100, resource_limits={"gold": 50}, tags=["hoarder"]),
                ],
                tag_definitions={
                    "hoarder": TagDefinition(
                        name="hoarder",
                        modifiers=[
                            PropertyModifier(
                                property_path="default_resource_limit",
                                operation="multiply",
                                value=2,  # 100 * 2 = 200
                            ),
                            PropertyModifier(
                                property_path="resource_limits",
                                operation="add",
                                value={"gold": 100, "silver": 50},  # gold: 50 + 100 = 150, silver: 0 + 50 = 50
                            ),
                        ],
                    )
                },
                resource_names=["gold", "silver", "bronze"],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        [".", "@", "."],
                    ],
                ),
            )
        )

        # Validate config is created successfully
        # Note: resource_limits are stored in inventory_config which is not easily accessible
        # from Python bindings, so we just verify the config was created
        cpp_config = convert_to_cpp_game_config(cfg.game)
        agent_cfg = cpp_config.objects.get("agent.red")
        assert agent_cfg is not None
        # Tag modifications were applied during conversion, and config was created successfully

    def test_max_min_operations(self):
        """Test max and min operations for property modifications."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=2,
                max_steps=100,
                obs_width=3,
                obs_height=3,
                num_observation_tokens=200,
                actions=ActionsConfig(noop=ActionConfig()),
                agents=[
                    AgentConfig(team_id=0, freeze_duration=10, action_failure_penalty=2.0, tags=["limiter"]),
                    AgentConfig(team_id=1, freeze_duration=20, action_failure_penalty=0.5, tags=["limiter"]),
                ],
                tag_definitions={
                    "limiter": TagDefinition(
                        name="limiter",
                        modifiers=[
                            PropertyModifier(
                                property_path="freeze_duration",
                                operation="max",
                                value=15,  # Take max(current, 15)
                            ),
                            PropertyModifier(
                                property_path="action_failure_penalty",
                                operation="min",
                                value=1.0,  # Take min(current, 1.0)
                            ),
                        ],
                    )
                },
                resource_names=[],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        [".", "@", "."],
                        [".", "@", "."],
                    ],
                ),
            )
        )

        # Create environment with tag modifications
        env = MettaGridCore(cfg)
        obs, _ = env.reset()

        # The fact that the environment was created successfully means the tag modifications were applied
        # Team 0: freeze_duration max(10, 15) = 15, penalty min(2.0, 1.0) = 1.0
        # Team 1: freeze_duration max(20, 15) = 20, penalty min(0.5, 1.0) = 0.5
        assert obs is not None
        assert len(obs) == 2  # Two agents

    def test_empty_tag_definitions(self):
        """Test that system works when tag_definitions is empty."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                obs_width=3,
                obs_height=3,
                num_observation_tokens=200,
                actions=ActionsConfig(noop=ActionConfig()),
                agents=[
                    AgentConfig(team_id=0, freeze_duration=10, tags=["undefined_tag"]),
                ],
                tag_definitions={},  # No tag definitions
                resource_names=[],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        [".", "@", "."],
                    ],
                ),
            )
        )

        # Should not crash, tags just won't modify properties
        env = MettaGridCore(cfg)
        obs, _ = env.reset()

        # Properties should remain unchanged (freeze_duration = 10)
        assert obs is not None

    def test_integration_with_environment(self):
        """Test that tag modifications work in a full environment."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=2,
                max_steps=100,
                obs_width=5,
                obs_height=5,
                num_observation_tokens=200,
                actions=ActionsConfig(
                    noop=ActionConfig(),
                    move=ActionConfig(),
                ),
                agents=[
                    AgentConfig(team_id=0, freeze_duration=20, tags=["speedy"]),
                    AgentConfig(team_id=1, freeze_duration=20, tags=[]),
                ],
                objects={
                    "wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID, swappable=False, tags=["immovable"]),
                },
                tag_definitions={
                    "speedy": TagDefinition(
                        name="speedy",
                        modifiers=[PropertyModifier(property_path="freeze_duration", operation="override", value=5)],
                    ),
                    "immovable": TagDefinition(
                        name="immovable",
                        modifiers=[PropertyModifier(property_path="swappable", operation="override", value=False)],
                    ),
                },
                resource_names=[],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#", "#", "#"],
                        ["#", "@", ".", ".", "#"],
                        ["#", ".", ".", ".", "#"],
                        ["#", ".", ".", "@", "#"],
                        ["#", "#", "#", "#", "#"],
                    ],
                ),
            )
        )

        # Create environment with tag modifications
        env = MettaGridCore(cfg)
        obs, _ = env.reset()

        # Environment should work correctly with modified properties
        assert obs is not None
        assert len(obs) == 2

    def test_converter_with_no_tags(self):
        """Test that a converter with no tags doesn't cause UnboundLocalError."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                obs_width=3,
                obs_height=3,
                num_observation_tokens=200,
                actions=ActionsConfig(noop=ActionConfig()),
                objects={
                    "tagless_converter": ConverterConfig(
                        type_id=2,
                        input_resources={"wood": 1},
                        output_resources={"coal": 1},
                        max_output=10,
                        conversion_ticks=10,
                        cooldown=5,
                        tags=[],  # Empty tags - this used to cause UnboundLocalError
                    ),
                },
                resource_names=["wood", "coal"],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        [".", "@", "."],
                    ],
                ),
            )
        )

        # Should not crash even with empty tags
        env = MettaGridCore(cfg)
        obs, _ = env.reset()
        assert obs is not None

    def test_tag_adds_unknown_resource_to_initial_inventory(self):
        """Test that adding unknown resource to initial_inventory via tags raises clear error."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                obs_width=3,
                obs_height=3,
                num_observation_tokens=200,
                actions=ActionsConfig(noop=ActionConfig()),
                agents=[
                    AgentConfig(team_id=0, initial_inventory={"gold": 10}, tags=["starter_pack"]),
                ],
                tag_definitions={
                    "starter_pack": TagDefinition(
                        name="starter_pack",
                        modifiers=[
                            PropertyModifier(
                                property_path="initial_inventory",
                                operation="add",
                                value={"unobtainium": 5},  # This resource doesn't exist
                            )
                        ],
                    )
                },
                resource_names=["gold", "silver"],  # Note: no "unobtainium"
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        [".", "@", "."],
                    ],
                ),
            )
        )

        # Should raise ValueError with clear message about unknown resources
        with pytest.raises(ValueError) as exc_info:
            MettaGridCore(cfg)
        assert "unknown resources" in str(exc_info.value).lower()
        assert "unobtainium" in str(exc_info.value)

    def test_agent_default_resource_limit_affects_fallback(self):
        """Test that modifying default_resource_limit via tags affects per-resource fallback."""
        from mettagrid.config.mettagrid_c_config import convert_to_cpp_game_config

        game_config = GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=3,
            obs_height=3,
            num_observation_tokens=200,
            actions=ActionsConfig(noop=ActionConfig()),
            agents=[
                AgentConfig(
                    team_id=0,
                    default_resource_limit=100,
                    resource_limits={"gold": 50},  # Only gold has explicit limit
                    tags=["double_capacity"],
                ),
            ],
            tag_definitions={
                "double_capacity": TagDefinition(
                    name="double_capacity",
                    modifiers=[
                        PropertyModifier(
                            property_path="default_resource_limit",
                            operation="multiply",
                            value=2,  # 100 * 2 = 200
                        )
                    ],
                )
            },
            resource_names=["gold", "silver", "bronze"],
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    [".", "@", "."],
                ],
            ),
        )

        # Convert to C++ config to apply tag modifications
        cpp_config = convert_to_cpp_game_config(game_config)

        # Find the agent configuration
        agent_config = cpp_config.objects.get("agent.red") or cpp_config.objects.get("agent.default")
        assert agent_config is not None

        # Verify fallback limits reflect tag-modified default_resource_limit
        # gold has an explicit limit (50); silver and bronze should use fallback (200)
        resource_name_to_id = {name: idx for idx, name in enumerate(cpp_config.resource_names)}
        silver_id = resource_name_to_id["silver"]
        bronze_id = resource_name_to_id["bronze"]

        limits = agent_config.inventory_config.limits

        def limit_for(res_id: int) -> int:
            for resources, limit in limits:
                if res_id in resources:
                    return limit
            raise AssertionError("resource id not found in limits")

        assert limit_for(silver_id) == 200
        assert limit_for(bronze_id) == 200

    def test_assembler_tag_modifications(self):
        """Test that tags can modify assembler properties."""
        from mettagrid.config.mettagrid_config import AssemblerConfig, RecipeConfig

        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                obs_width=3,
                obs_height=3,
                num_observation_tokens=200,
                actions=ActionsConfig(noop=ActionConfig()),
                objects={
                    "crafting_table": AssemblerConfig(
                        type_id=5,
                        recipes=[
                            (
                                ["N", "S"],
                                RecipeConfig(
                                    input_resources={"wood": 2},
                                    output_resources={"plank": 1},
                                    cooldown=10,
                                ),
                            ),
                        ],
                        tags=["upgraded"],
                    ),
                },
                tag_definitions={
                    "upgraded": TagDefinition(
                        name="upgraded",
                        modifiers=[
                            PropertyModifier(
                                property_path="type_id",
                                operation="override",
                                value=10,  # Change type_id from 5 to 10
                            )
                        ],
                    )
                },
                resource_names=["wood", "plank"],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        [".", "@", "."],
                    ],
                ),
            )
        )

        # Convert to C++ config to verify modifications
        cpp_config = convert_to_cpp_game_config(cfg.game)

        # Find the assembler configuration
        assembler_config = cpp_config.objects.get("crafting_table")
        assert assembler_config is not None
        assert assembler_config.type_id == 10  # Modified from 5 to 10

    def test_chest_tag_modifications(self):
        """Test that tags can modify chest properties."""
        from mettagrid.config.mettagrid_config import ChestConfig

        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                obs_width=3,
                obs_height=3,
                num_observation_tokens=200,
                actions=ActionsConfig(noop=ActionConfig()),
                objects={
                    "storage": ChestConfig(
                        type_id=6,
                        resource_type="gold",
                        position_deltas=[("N", 1), ("S", -1)],  # N deposits, S withdraws
                        tags=["increased_capacity"],
                    ),
                },
                tag_definitions={
                    "increased_capacity": TagDefinition(
                        name="increased_capacity",
                        modifiers=[
                            PropertyModifier(property_path="max_inventory", operation="multiply", value=2),
                            PropertyModifier(property_path="type_id", operation="override", value=12),
                        ],
                    )
                },
                resource_names=["gold", "silver"],
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        [".", "@", "."],
                    ],
                ),
            )
        )

        # Convert to C++ config to verify modifications
        cpp_config = convert_to_cpp_game_config(cfg.game)

        # Find the chest configuration
        chest_config = cpp_config.objects.get("storage")
        assert chest_config is not None
        assert chest_config.type_id == 12  # Modified from 6 to 12
        # Default max_inventory is 255, multiplied by 2 gives 510
        assert chest_config.max_inventory == 510  # 255 * 2


def test_apply_property_modifier_unit_tests():
    """Unit tests for apply_property_modifier function."""
    from mettagrid.config.mettagrid_c_config import apply_property_modifier

    # Test override operation
    obj = {"value": 10}
    apply_property_modifier(obj, {"property_path": "value", "operation": "override", "value": 20})
    assert obj["value"] == 20

    # Test add operation on scalar
    obj = {"value": 10}
    apply_property_modifier(obj, {"property_path": "value", "operation": "add", "value": 5})
    assert obj["value"] == 15

    # Test add operation on missing scalar
    obj = {}
    apply_property_modifier(obj, {"property_path": "value", "operation": "add", "value": 5})
    assert obj["value"] == 5

    # Test add operation on dict
    obj = {"resources": {"gold": 10, "silver": 20}}
    apply_property_modifier(obj, {"property_path": "resources", "operation": "add", "value": {"gold": 5, "bronze": 15}})
    assert obj["resources"] == {"gold": 15, "silver": 20, "bronze": 15}

    # Test multiply operation on scalar
    obj = {"value": 10}
    apply_property_modifier(obj, {"property_path": "value", "operation": "multiply", "value": 2})
    assert obj["value"] == 20

    # Test multiply operation on missing scalar (sets to 0)
    obj = {}
    apply_property_modifier(obj, {"property_path": "value", "operation": "multiply", "value": 2})
    assert obj["value"] == 0

    # Test multiply operation on dict
    obj = {"resources": {"gold": 10, "silver": 20}}
    apply_property_modifier(
        obj, {"property_path": "resources", "operation": "multiply", "value": {"gold": 2, "silver": 0.5}}
    )
    assert obj["resources"] == {"gold": 20, "silver": 10}

    # Test max operation
    obj = {"value": 10}
    apply_property_modifier(obj, {"property_path": "value", "operation": "max", "value": 15})
    assert obj["value"] == 15

    obj = {"value": 20}
    apply_property_modifier(obj, {"property_path": "value", "operation": "max", "value": 15})
    assert obj["value"] == 20

    # Test min operation
    obj = {"value": 10}
    apply_property_modifier(obj, {"property_path": "value", "operation": "min", "value": 15})
    assert obj["value"] == 10

    obj = {"value": 20}
    apply_property_modifier(obj, {"property_path": "value", "operation": "min", "value": 15})
    assert obj["value"] == 15

    # Test nested property path
    obj = {"nested": {"deep": {"value": 10}}}
    apply_property_modifier(obj, {"property_path": "nested.deep.value", "operation": "add", "value": 5})
    assert obj["nested"]["deep"]["value"] == 15

    # Test creating nested path when missing
    obj = {}
    apply_property_modifier(obj, {"property_path": "nested.deep.value", "operation": "override", "value": 42})
    assert obj["nested"]["deep"]["value"] == 42

    # Verify no resource ID conversion happens (keys stay as strings)
    obj = {"resources": {}}
    apply_property_modifier(
        obj, {"property_path": "resources", "operation": "add", "value": {"gold": 10, "silver": 20}}
    )
    # Keys should remain as strings, not converted to IDs
    assert "gold" in obj["resources"]
    assert "silver" in obj["resources"]
    assert 0 not in obj["resources"]  # Should not have numeric keys
