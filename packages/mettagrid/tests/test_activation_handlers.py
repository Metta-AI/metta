"""Tests for activation handler configuration and conversion."""

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    ActivationHandlerConfig,
    FactionConfig,
    FilterConfig,
    GameConfig,
    MettaGridConfig,
    MoveActionConfig,
    MutationConfig,
    NoopActionConfig,
    ObsConfig,
    WallConfig,
)
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.simulator import Simulation


class TestActivationHandlerConfig:
    """Tests for ActivationHandlerConfig Python model."""

    def test_handler_config_basic(self):
        """Test basic handler config creation."""
        handler = ActivationHandlerConfig(
            name="heal_handler",
            filters=[FilterConfig(type="vibe", entity="actor", vibe="weapon")],
            mutations=[MutationConfig(type="resource_delta", entity="target", resource_deltas={"energy": 10})],
        )
        assert handler.name == "heal_handler"
        assert len(handler.filters) == 1
        assert len(handler.mutations) == 1

    def test_filter_config_vibe(self):
        """Test vibe filter config."""
        filter_cfg = FilterConfig(type="vibe", entity="actor", vibe="weapon")
        assert filter_cfg.type == "vibe"
        assert filter_cfg.entity == "actor"
        assert filter_cfg.vibe == "weapon"

    def test_filter_config_resource(self):
        """Test resource filter config."""
        filter_cfg = FilterConfig(type="resource", entity="target", resource="gold", min_amount=5)
        assert filter_cfg.type == "resource"
        assert filter_cfg.entity == "target"
        assert filter_cfg.resource == "gold"
        assert filter_cfg.min_amount == 5

    def test_filter_config_alignment(self):
        """Test alignment filter config."""
        filter_cfg = FilterConfig(type="alignment", alignment_condition="same_faction")
        assert filter_cfg.type == "alignment"
        assert filter_cfg.alignment_condition == "same_faction"

    def test_filter_config_tag(self):
        """Test tag filter config."""
        filter_cfg = FilterConfig(type="tag", entity="target", tag="healer")
        assert filter_cfg.type == "tag"
        assert filter_cfg.entity == "target"
        assert filter_cfg.tag == "healer"

    def test_mutation_config_resource_delta(self):
        """Test resource delta mutation."""
        mutation = MutationConfig(type="resource_delta", entity="target", resource_deltas={"energy": 10, "gold": -5})
        assert mutation.type == "resource_delta"
        assert mutation.entity == "target"
        assert mutation.resource_deltas == {"energy": 10, "gold": -5}

    def test_mutation_config_resource_transfer(self):
        """Test resource transfer mutation."""
        mutation = MutationConfig(
            type="resource_transfer",
            transfer_source="actor",
            transfer_target="target",
            resource_deltas={"gold": 5},
        )
        assert mutation.type == "resource_transfer"
        assert mutation.transfer_source == "actor"
        assert mutation.transfer_target == "target"
        assert mutation.resource_deltas == {"gold": 5}

    def test_mutation_config_alignment(self):
        """Test alignment mutation."""
        mutation = MutationConfig(type="alignment", align_to_actor=True)
        assert mutation.type == "alignment"
        assert mutation.align_to_actor is True

    def test_mutation_config_freeze(self):
        """Test freeze mutation."""
        mutation = MutationConfig(type="freeze", freeze_duration=10)
        assert mutation.type == "freeze"
        assert mutation.freeze_duration == 10

    def test_mutation_config_attack(self):
        """Test attack mutation."""
        mutation = MutationConfig(type="attack", attack_damage=5)
        assert mutation.type == "attack"
        assert mutation.attack_damage == 5

    def test_wall_with_activation_handler(self):
        """Test WallConfig with activation handler."""
        wall = WallConfig(
            name="healing_station",
            activation_handlers=[
                ActivationHandlerConfig(
                    name="heal",
                    filters=[FilterConfig(type="tag", entity="target", tag="agent")],
                    mutations=[MutationConfig(type="resource_delta", entity="target", resource_deltas={"energy": 5})],
                )
            ],
        )
        assert len(wall.activation_handlers) == 1
        assert wall.activation_handlers[0].name == "heal"

    def test_wall_with_multiple_handlers(self):
        """Test WallConfig with multiple activation handlers."""
        wall = WallConfig(
            name="multi_station",
            activation_handlers=[
                ActivationHandlerConfig(
                    name="heal",
                    filters=[],
                    mutations=[MutationConfig(type="resource_delta", entity="target", resource_deltas={"energy": 5})],
                ),
                ActivationHandlerConfig(
                    name="damage",
                    filters=[FilterConfig(type="vibe", entity="actor", vibe="weapon")],
                    mutations=[MutationConfig(type="attack", attack_damage=3)],
                ),
            ],
        )
        assert len(wall.activation_handlers) == 2


class TestActivationHandlerIntegration:
    """Integration tests for activation handlers with Simulation."""

    def test_handler_creates_simulation(self):
        """Test that a game config with activation handlers creates a Simulation."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                resource_names=["energy", "gold"],
                obs=ObsConfig(width=5, height=5, num_tokens=100),
                actions=ActionsConfig(
                    noop=NoopActionConfig(),
                    move=MoveActionConfig(enabled=True),
                ),
                objects={
                    "wall": WallConfig(),
                    "heal_station": WallConfig(
                        name="heal_station",
                        activation_handlers=[
                            ActivationHandlerConfig(
                                name="heal",
                                filters=[],
                                mutations=[
                                    MutationConfig(
                                        type="resource_delta", entity="target", resource_deltas={"energy": 5}
                                    )
                                ],
                            )
                        ],
                    ),
                },
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#", "#", "#"],
                        ["#", "H", ".", ".", "#"],
                        ["#", ".", "@", ".", "#"],
                        ["#", ".", ".", ".", "#"],
                        ["#", "#", "#", "#", "#"],
                    ],
                    char_to_map_name={
                        "#": "wall",
                        "@": "agent.agent",
                        ".": "empty",
                        "H": "heal_station",
                    },
                ),
            )
        )

        # Should not raise
        sim = Simulation(cfg)
        assert sim is not None

    def test_handler_with_vibe_filter_integration(self):
        """Test handler with vibe filter creates successfully."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                resource_names=["energy"],
                obs=ObsConfig(width=5, height=5, num_tokens=100),
                actions=ActionsConfig(
                    noop=NoopActionConfig(),
                    move=MoveActionConfig(enabled=True),
                ),
                objects={
                    "wall": WallConfig(),
                    "vibe_station": WallConfig(
                        name="vibe_station",
                        activation_handlers=[
                            ActivationHandlerConfig(
                                name="vibe_heal",
                                filters=[FilterConfig(type="vibe", entity="actor", vibe="weapon")],
                                mutations=[
                                    MutationConfig(
                                        type="resource_delta", entity="target", resource_deltas={"energy": 10}
                                    )
                                ],
                            )
                        ],
                    ),
                },
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#", "#", "#"],
                        ["#", "V", ".", ".", "#"],
                        ["#", ".", "@", ".", "#"],
                        ["#", ".", ".", ".", "#"],
                        ["#", "#", "#", "#", "#"],
                    ],
                    char_to_map_name={
                        "#": "wall",
                        "@": "agent.agent",
                        ".": "empty",
                        "V": "vibe_station",
                    },
                ),
            )
        )

        sim = Simulation(cfg)
        assert sim is not None

    def test_handler_with_alignment_filter_integration(self):
        """Test handler with alignment filter creates successfully."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                resource_names=["energy"],
                obs=ObsConfig(width=5, height=5, num_tokens=100),
                actions=ActionsConfig(
                    noop=NoopActionConfig(),
                    move=MoveActionConfig(enabled=True),
                ),
                factions=[FactionConfig(name="allies")],
                objects={
                    "wall": WallConfig(),
                    "faction_station": WallConfig(
                        name="faction_station",
                        faction="allies",
                        activation_handlers=[
                            ActivationHandlerConfig(
                                name="faction_heal",
                                filters=[FilterConfig(type="alignment", alignment_condition="same_faction")],
                                mutations=[
                                    MutationConfig(
                                        type="resource_delta", entity="target", resource_deltas={"energy": 5}
                                    )
                                ],
                            )
                        ],
                    ),
                },
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#", "#", "#"],
                        ["#", "F", ".", ".", "#"],
                        ["#", ".", "@", ".", "#"],
                        ["#", ".", ".", ".", "#"],
                        ["#", "#", "#", "#", "#"],
                    ],
                    char_to_map_name={
                        "#": "wall",
                        "@": "agent.agent",
                        ".": "empty",
                        "F": "faction_station",
                    },
                ),
            )
        )

        sim = Simulation(cfg)
        assert sim is not None

    def test_handler_with_complex_mutations(self):
        """Test handler with multiple mutation types."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                resource_names=["energy", "gold"],
                obs=ObsConfig(width=5, height=5, num_tokens=100),
                actions=ActionsConfig(
                    noop=NoopActionConfig(),
                    move=MoveActionConfig(enabled=True),
                ),
                objects={
                    "wall": WallConfig(),
                    "complex_station": WallConfig(
                        name="complex_station",
                        activation_handlers=[
                            ActivationHandlerConfig(
                                name="complex_handler",
                                filters=[FilterConfig(type="resource", entity="actor", resource="gold", min_amount=1)],
                                mutations=[
                                    MutationConfig(
                                        type="resource_delta",
                                        entity="target",
                                        resource_deltas={"energy": 10, "gold": -1},
                                    ),
                                    MutationConfig(type="freeze", freeze_duration=5),
                                ],
                            )
                        ],
                    ),
                },
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#", "#", "#"],
                        ["#", "C", ".", ".", "#"],
                        ["#", ".", "@", ".", "#"],
                        ["#", ".", ".", ".", "#"],
                        ["#", "#", "#", "#", "#"],
                    ],
                    char_to_map_name={
                        "#": "wall",
                        "@": "agent.agent",
                        ".": "empty",
                        "C": "complex_station",
                    },
                ),
            )
        )

        sim = Simulation(cfg)
        assert sim is not None
