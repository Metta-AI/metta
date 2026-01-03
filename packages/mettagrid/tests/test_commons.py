"""Test commons (shared inventory) functionality for mettagrid."""

from mettagrid.config.mettagrid_c_config import convert_to_cpp_game_config
from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AssemblerConfig,
    ChestConfig,
    CommonsConfig,
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


class TestCommonsConfig:
    """Test CommonsConfig creation and conversion."""

    def test_commons_config_basic(self):
        """Test that CommonsConfig can be created with basic fields."""
        cfg = CommonsConfig(
            name="shared_storage",
            inventory=InventoryConfig(
                initial={"gold": 100},
                limits={"gold": ResourceLimitsConfig(limit=1000, resources=["gold"])},
            ),
        )
        assert cfg.name == "shared_storage"
        assert cfg.inventory.initial["gold"] == 100

    def test_commons_tag_auto_added(self):
        """Test that commons field adds the commons tag automatically."""
        cfg = WallConfig(name="my_wall", commons="shared")
        assert "commons:shared" in cfg.tags

    def test_commons_tag_not_duplicated(self):
        """Test that commons tag is not duplicated if already present."""
        cfg = WallConfig(name="my_wall", tags=["commons:shared"], commons="shared")
        assert cfg.tags.count("commons:shared") == 1

    def test_game_config_with_commons(self):
        """Test that GameConfig accepts commons list."""
        game_config = GameConfig(
            num_agents=1,
            commons=[
                CommonsConfig(name="team_storage", inventory=InventoryConfig(initial={"gold": 50})),
            ],
            resource_names=["gold"],
        )
        assert len(game_config.commons) == 1
        assert game_config.commons[0].name == "team_storage"


class TestCommonsConversion:
    """Test Python to C++ commons conversion."""

    def test_commons_cpp_conversion(self):
        """Test that commons configs are properly converted to C++."""
        game_config = GameConfig(
            num_agents=1,
            resource_names=["gold", "silver"],
            commons=[
                CommonsConfig(
                    name="vault",
                    inventory=InventoryConfig(
                        initial={"gold": 100, "silver": 50},
                        limits={"precious": ResourceLimitsConfig(limit=500, resources=["gold", "silver"])},
                    ),
                ),
            ],
        )

        cpp_config = convert_to_cpp_game_config(game_config)

        # Check that commons was converted
        assert "vault" in cpp_config.commons
        vault_config = cpp_config.commons["vault"]
        assert vault_config.name == "vault"

        # Check initial inventory was converted (resource IDs, not names)
        gold_id = game_config.resource_names.index("gold")
        silver_id = game_config.resource_names.index("silver")
        assert vault_config.initial_inventory[gold_id] == 100
        assert vault_config.initial_inventory[silver_id] == 50


class TestCommonsIntegration:
    """Test commons integration with the simulation."""

    def test_commons_with_objects(self):
        """Test that objects can be associated with a commons."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                resource_names=["gold"],
                actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
                commons=[
                    CommonsConfig(
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
                        commons="team_vault",  # Associate with commons
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

        # Verify commons tag was added to chest
        assert "commons:team_vault" in cfg.game.objects["chest"].tags

        # Create simulation - this verifies the C++ side accepts our config
        sim = Simulation(cfg)
        assert sim is not None

        # The simulation should start successfully
        obs = sim._c_sim.observations()
        assert obs is not None

    def test_multiple_commons(self):
        """Test that multiple commons can be configured."""
        game_config = GameConfig(
            num_agents=2,
            resource_names=["gold", "silver"],
            commons=[
                CommonsConfig(name="team_red_vault", inventory=InventoryConfig(initial={"gold": 50})),
                CommonsConfig(name="team_blue_vault", inventory=InventoryConfig(initial={"silver": 50})),
            ],
        )

        cpp_config = convert_to_cpp_game_config(game_config)

        assert len(cpp_config.commons) == 2
        assert "team_red_vault" in cpp_config.commons
        assert "team_blue_vault" in cpp_config.commons

    def test_commons_with_assembler(self):
        """Test that assemblers can be associated with a commons."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                resource_names=["ore", "metal"],
                actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
                commons=[
                    CommonsConfig(
                        name="factory_storage",
                        inventory=InventoryConfig(initial={"ore": 100}),
                    ),
                ],
                objects={
                    "wall": WallConfig(),
                    "smelter": AssemblerConfig(
                        name="smelter",
                        commons="factory_storage",
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

        # Verify commons tag was added to assembler
        assert "commons:factory_storage" in cfg.game.objects["smelter"].tags

        # Create simulation
        sim = Simulation(cfg)
        assert sim is not None


class TestCommonsTagMapping:
    """Test that commons tags are properly mapped in the tag system."""

    def test_commons_tag_in_tag_map(self):
        """Test that commons tags appear in the tag_id_map."""
        game_config = GameConfig(
            num_agents=1,
            resource_names=["gold"],
            commons=[CommonsConfig(name="vault", inventory=InventoryConfig())],
            objects={
                "wall": WallConfig(commons="vault"),
            },
        )

        cpp_config = convert_to_cpp_game_config(game_config)

        # Check that commons:vault tag is in the tag_id_map
        tag_names = list(cpp_config.tag_id_map.values())
        assert "commons:vault" in tag_names

    def test_multiple_objects_same_commons(self):
        """Test that multiple objects can share the same commons tag."""
        game_config = GameConfig(
            num_agents=1,
            resource_names=["gold"],
            commons=[CommonsConfig(name="shared", inventory=InventoryConfig())],
            objects={
                "wall1": WallConfig(name="wall1", commons="shared"),
                "wall2": WallConfig(name="wall2", commons="shared"),
            },
        )

        cpp_config = convert_to_cpp_game_config(game_config)

        # Both walls should have the same commons tag
        # The tag should appear only once in the map
        tag_names = list(cpp_config.tag_id_map.values())
        assert tag_names.count("commons:shared") == 1
