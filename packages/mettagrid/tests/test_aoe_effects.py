"""Tests for Area of Effect (AOE) system.

This tests the AOE system functionality where:
- Objects can emit AOE effects that register with the grid
- Effects are tracked per-cell with source information
- Effects can be filtered by target tags and faction membership
"""

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AOEEffectConfig,
    GameConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    WallConfig,
)
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.simulator import Simulation


class TestAOEConfig:
    """Test AOEEffectConfig creation and conversion."""

    def test_aoe_config_basic(self):
        """Test that AOEEffectConfig can be created with basic fields."""
        cfg = AOEEffectConfig(
            range=2,
            resource_deltas={"energy": 5},
        )
        assert cfg.range == 2
        assert cfg.resource_deltas["energy"] == 5

    def test_aoe_config_defaults(self):
        """Test AOEEffectConfig default values."""
        cfg = AOEEffectConfig()
        assert cfg.range == 1
        assert cfg.resource_deltas == {}
        assert cfg.target_tags == []
        assert cfg.same_faction_only is False
        assert cfg.different_faction_only is False

    def test_wall_with_aoe_config(self):
        """Test that WallConfig can include AOE effects."""
        cfg = WallConfig(
            name="energy_emitter",
            aoes=[
                AOEEffectConfig(
                    range=2,
                    resource_deltas={"energy": 10},
                ),
            ],
        )
        assert len(cfg.aoes) == 1
        assert cfg.aoes[0].range == 2
        assert cfg.aoes[0].resource_deltas["energy"] == 10

    def test_wall_with_multiple_aoes(self):
        """Test that WallConfig can have multiple AOE effects."""
        cfg = WallConfig(
            name="dual_emitter",
            aoes=[
                AOEEffectConfig(range=1, resource_deltas={"energy": 5}),
                AOEEffectConfig(range=3, resource_deltas={"heart": -1}),
            ],
        )
        assert len(cfg.aoes) == 2


class TestAOEIntegration:
    """Test AOE integration with the simulation."""

    def test_aoe_wall_creates_simulation(self):
        """Test that a wall with AOE config creates a valid simulation."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                resource_names=["energy", "heart"],
                obs=ObsConfig(width=5, height=5, num_tokens=100),
                actions=ActionsConfig(
                    noop=NoopActionConfig(),
                    move=MoveActionConfig(enabled=True),
                ),
                objects={
                    "wall": WallConfig(),
                    "emitter": WallConfig(
                        name="emitter",
                        aoes=[AOEEffectConfig(range=2, resource_deltas={"energy": 5})],
                    ),
                },
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#", "#", "#"],
                        ["#", "E", ".", ".", "#"],
                        ["#", ".", "@", ".", "#"],
                        ["#", ".", ".", ".", "#"],
                        ["#", "#", "#", "#", "#"],
                    ],
                    char_to_map_name={
                        "#": "wall",
                        "@": "agent.agent",
                        ".": "empty",
                        "E": "emitter",
                    },
                ),
            )
        )

        # Create simulation - this verifies the C++ side accepts our config
        sim = Simulation(cfg)
        assert sim is not None

        # The simulation should start successfully
        obs = sim._c_sim.observations()
        assert obs is not None

    def test_aoe_config_with_target_tags(self):
        """Test AOE with target tag filtering."""
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
                    "emitter": WallConfig(
                        name="emitter",
                        aoes=[
                            AOEEffectConfig(
                                range=2,
                                resource_deltas={"energy": 5},
                                target_tags=["friendly"],  # Only affect objects with 'friendly' tag
                            )
                        ],
                    ),
                },
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#"],
                        ["#", "@", "#"],
                        ["#", "#", "#"],
                    ],
                    char_to_map_name={
                        "#": "wall",
                        "@": "agent.agent",
                        ".": "empty",
                    },
                ),
            )
        )

        # Should create successfully even with target_tags filter
        sim = Simulation(cfg)
        assert sim is not None

    def test_aoe_config_with_faction_filters(self):
        """Test AOE with faction-based filtering."""
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
                    "healer": WallConfig(
                        name="healer",
                        aoes=[
                            AOEEffectConfig(
                                range=2,
                                resource_deltas={"energy": 10},
                                same_faction_only=True,  # Only heal faction members
                            )
                        ],
                    ),
                },
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#"],
                        ["#", "@", "#"],
                        ["#", "#", "#"],
                    ],
                    char_to_map_name={
                        "#": "wall",
                        "@": "agent.agent",
                        ".": "empty",
                    },
                ),
            )
        )

        # Should create successfully with faction filter
        sim = Simulation(cfg)
        assert sim is not None
