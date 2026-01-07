"""Test collective (shared inventory) functionality for mettagrid."""

from mettagrid.config.mettagrid_c_config import convert_to_cpp_game_config
from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AssemblerConfig,
    ChestConfig,
    CollectiveConfig,
    GameConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ProtocolConfig,
    ResourceLimitsConfig,
    WallConfig,
)
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.mapgen.utils.ascii_grid import DEFAULT_CHAR_TO_NAME
from mettagrid.simulator import Simulation


class TestCollectiveConfig:
    """Test CollectiveConfig creation and conversion."""

    def test_collective_config_basic(self):
        """Test that CollectiveConfig can be created with basic fields."""
        cfg = CollectiveConfig(
            name="shared_storage",
            inventory=InventoryConfig(
                initial={"gold": 100},
                limits={"gold": ResourceLimitsConfig(limit=1000, resources=["gold"])},
            ),
        )
        assert cfg.name == "shared_storage"
        assert cfg.inventory.initial["gold"] == 100

    def test_collective_tag_auto_added(self):
        """Test that collective field adds the collective tag automatically."""
        cfg = WallConfig(name="my_wall", collective="shared")
        assert "collective:shared" in cfg.tags

    def test_collective_tag_not_duplicated(self):
        """Test that collective tag is not duplicated if already present."""
        cfg = WallConfig(name="my_wall", tags=["collective:shared"], collective="shared")
        assert cfg.tags.count("collective:shared") == 1

    def test_game_config_with_collectives(self):
        """Test that GameConfig accepts collectives list."""
        game_config = GameConfig(
            num_agents=1,
            collectives=[
                CollectiveConfig(name="team_storage", inventory=InventoryConfig(initial={"gold": 50})),
            ],
            resource_names=["gold"],
        )
        assert len(game_config.collectives) == 1
        assert game_config.collectives[0].name == "team_storage"


class TestCollectiveConversion:
    """Test Python to C++ collective conversion."""

    def test_collective_cpp_conversion(self):
        """Test that collective configs are properly converted to C++."""
        game_config = GameConfig(
            num_agents=1,
            resource_names=["gold", "silver"],
            collectives=[
                CollectiveConfig(
                    name="vault",
                    inventory=InventoryConfig(
                        initial={"gold": 100, "silver": 50},
                        limits={"precious": ResourceLimitsConfig(limit=500, resources=["gold", "silver"])},
                    ),
                ),
            ],
        )

        cpp_config = convert_to_cpp_game_config(game_config)

        # Check that collective was converted
        assert "vault" in cpp_config.collectives
        vault_config = cpp_config.collectives["vault"]
        assert vault_config.name == "vault"

        # Check initial inventory was converted (resource IDs, not names)
        gold_id = game_config.resource_names.index("gold")
        silver_id = game_config.resource_names.index("silver")
        assert vault_config.initial_inventory[gold_id] == 100
        assert vault_config.initial_inventory[silver_id] == 50


class TestCollectiveIntegration:
    """Test collective integration with the simulation."""

    def test_collective_with_objects(self):
        """Test that objects can be associated with a collective."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                resource_names=["gold"],
                actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
                collectives=[
                    CollectiveConfig(
                        name="team_vault",
                        inventory=InventoryConfig(
                            initial={"gold": 100},
                            limits={"gold": ResourceLimitsConfig(limit=1000, resources=["gold"])},
                        ),
                    ),
                ],
                objects={
                    "wall": WallConfig(),
                    "chest": ChestConfig(
                        name="team_chest",
                        collective="team_vault",  # Associate with collective
                        vibe_transfers={"up": {"gold": -10}},  # withdraw 10 gold
                    ),
                },
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#", "#", "#"],
                        ["#", ".", ".", ".", "#"],
                        ["#", ".", "C", ".", "#"],
                        ["#", ".", "@", ".", "#"],
                        ["#", "#", "#", "#", "#"],
                    ],
                    char_to_map_name={
                        "#": "wall",
                        "@": "agent.agent",
                        ".": "empty",
                        "C": "team_chest",
                    },
                ),
            )
        )

        # Verify collective tag was added to chest
        assert "collective:team_vault" in cfg.game.objects["chest"].tags

        # Create simulation - this verifies the C++ side accepts our config
        sim = Simulation(cfg)
        assert sim is not None

        # The simulation should start successfully
        obs = sim._c_sim.observations()
        assert obs is not None

    def test_multiple_collectives(self):
        """Test that multiple collectives can be configured."""
        game_config = GameConfig(
            num_agents=2,
            resource_names=["gold", "silver"],
            collectives=[
                CollectiveConfig(name="team_red_vault", inventory=InventoryConfig(initial={"gold": 50})),
                CollectiveConfig(name="team_blue_vault", inventory=InventoryConfig(initial={"silver": 50})),
            ],
        )

        cpp_config = convert_to_cpp_game_config(game_config)

        assert len(cpp_config.collectives) == 2
        assert "team_red_vault" in cpp_config.collectives
        assert "team_blue_vault" in cpp_config.collectives

    def test_collective_with_assembler(self):
        """Test that assemblers can be associated with a collective."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                resource_names=["ore", "metal"],
                actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
                collectives=[
                    CollectiveConfig(
                        name="factory_storage",
                        inventory=InventoryConfig(initial={"ore": 100}),
                    ),
                ],
                objects={
                    "wall": WallConfig(),
                    "smelter": AssemblerConfig(
                        name="smelter",
                        collective="factory_storage",
                        protocols=[
                            ProtocolConfig(input_resources={"ore": 1}, output_resources={"metal": 1}, cooldown=5)
                        ],
                    ),
                },
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#"],
                        ["#", "@", "#"],
                        ["#", "#", "#"],
                    ],
                    char_to_map_name=DEFAULT_CHAR_TO_NAME,
                ),
            )
        )

        # Verify collective tag was added to assembler
        assert "collective:factory_storage" in cfg.game.objects["smelter"].tags

        # Create simulation
        sim = Simulation(cfg)
        assert sim is not None


class TestCollectiveTagMapping:
    """Test that collective tags are properly mapped in the tag system."""

    def test_collective_tag_in_tag_map(self):
        """Test that collective tags appear in the tag_id_map."""
        game_config = GameConfig(
            num_agents=1,
            resource_names=["gold"],
            collectives=[CollectiveConfig(name="vault", inventory=InventoryConfig())],
            objects={
                "wall": WallConfig(collective="vault"),
            },
        )

        cpp_config = convert_to_cpp_game_config(game_config)

        # Check that collective:vault tag is in the tag_id_map
        tag_names = list(cpp_config.tag_id_map.values())
        assert "collective:vault" in tag_names

    def test_multiple_objects_same_collective(self):
        """Test that multiple objects can share the same collective tag."""
        game_config = GameConfig(
            num_agents=1,
            resource_names=["gold"],
            collectives=[CollectiveConfig(name="shared", inventory=InventoryConfig())],
            objects={
                "wall1": WallConfig(name="wall1", collective="shared"),
                "wall2": WallConfig(name="wall2", collective="shared"),
            },
        )

        cpp_config = convert_to_cpp_game_config(game_config)

        # Both walls should have the same collective tag
        # The tag should appear only once in the map
        tag_names = list(cpp_config.tag_id_map.values())
        assert tag_names.count("collective:shared") == 1
