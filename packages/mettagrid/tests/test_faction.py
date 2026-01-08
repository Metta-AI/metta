"""Test faction (shared inventory) functionality for mettagrid."""

from mettagrid.config.mettagrid_c_config import convert_to_cpp_game_config
from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AssemblerConfig,
    ChestConfig,
    FactionConfig,
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


class TestFactionConfig:
    """Test FactionConfig creation and conversion."""

    def test_faction_config_basic(self):
        """Test that FactionConfig can be created with basic fields."""
        cfg = FactionConfig(
            name="shared_storage",
            inventory=InventoryConfig(
                initial={"gold": 100},
                limits={"gold": ResourceLimitsConfig(limit=1000, resources=["gold"])},
            ),
        )
        assert cfg.name == "shared_storage"
        assert cfg.inventory.initial["gold"] == 100

    def test_faction_tag_auto_added(self):
        """Test that faction field adds the faction tag automatically."""
        cfg = WallConfig(name="my_wall", faction="shared")
        assert "faction:shared" in cfg.tags

    def test_faction_tag_not_duplicated(self):
        """Test that faction tag is not duplicated if already present."""
        cfg = WallConfig(name="my_wall", tags=["faction:shared"], faction="shared")
        assert cfg.tags.count("faction:shared") == 1

    def test_game_config_with_factions(self):
        """Test that GameConfig accepts factions list."""
        game_config = GameConfig(
            num_agents=1,
            factions=[
                FactionConfig(name="team_storage", inventory=InventoryConfig(initial={"gold": 50})),
            ],
            resource_names=["gold"],
        )
        assert len(game_config.factions) == 1
        assert game_config.factions[0].name == "team_storage"


class TestFactionConversion:
    """Test Python to C++ faction conversion."""

    def test_faction_cpp_conversion(self):
        """Test that faction configs are properly converted to C++."""
        game_config = GameConfig(
            num_agents=1,
            resource_names=["gold", "silver"],
            factions=[
                FactionConfig(
                    name="vault",
                    inventory=InventoryConfig(
                        initial={"gold": 100, "silver": 50},
                        limits={"precious": ResourceLimitsConfig(limit=500, resources=["gold", "silver"])},
                    ),
                ),
            ],
        )

        cpp_config = convert_to_cpp_game_config(game_config)

        # Check that faction was converted
        assert "vault" in cpp_config.factions
        vault_config = cpp_config.factions["vault"]
        assert vault_config.name == "vault"

        # Check initial inventory was converted (resource IDs, not names)
        gold_id = game_config.resource_names.index("gold")
        silver_id = game_config.resource_names.index("silver")
        assert vault_config.initial_inventory[gold_id] == 100
        assert vault_config.initial_inventory[silver_id] == 50


class TestFactionIntegration:
    """Test faction integration with the simulation."""

    def test_faction_with_objects(self):
        """Test that objects can be associated with a faction."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                resource_names=["gold"],
                actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
                factions=[
                    FactionConfig(
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
                        faction="team_vault",  # Associate with faction
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

        # Verify faction tag was added to chest
        assert "faction:team_vault" in cfg.game.objects["chest"].tags

        # Create simulation - this verifies the C++ side accepts our config
        sim = Simulation(cfg)
        assert sim is not None

        # The simulation should start successfully
        obs = sim._c_sim.observations()
        assert obs is not None

    def test_multiple_factions(self):
        """Test that multiple factions can be configured."""
        game_config = GameConfig(
            num_agents=2,
            resource_names=["gold", "silver"],
            factions=[
                FactionConfig(name="team_red_vault", inventory=InventoryConfig(initial={"gold": 50})),
                FactionConfig(name="team_blue_vault", inventory=InventoryConfig(initial={"silver": 50})),
            ],
        )

        cpp_config = convert_to_cpp_game_config(game_config)

        assert len(cpp_config.factions) == 2
        assert "team_red_vault" in cpp_config.factions
        assert "team_blue_vault" in cpp_config.factions

    def test_faction_with_assembler(self):
        """Test that assemblers can be associated with a faction."""
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                resource_names=["ore", "metal"],
                actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
                factions=[
                    FactionConfig(
                        name="factory_storage",
                        inventory=InventoryConfig(initial={"ore": 100}),
                    ),
                ],
                objects={
                    "wall": WallConfig(),
                    "smelter": AssemblerConfig(
                        name="smelter",
                        faction="factory_storage",
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

        # Verify faction tag was added to assembler
        assert "faction:factory_storage" in cfg.game.objects["smelter"].tags

        # Create simulation
        sim = Simulation(cfg)
        assert sim is not None


class TestFactionTagMapping:
    """Test that faction tags are properly mapped in the tag system."""

    def test_faction_tag_in_tag_map(self):
        """Test that faction tags appear in the tag_id_map."""
        game_config = GameConfig(
            num_agents=1,
            resource_names=["gold"],
            factions=[FactionConfig(name="vault", inventory=InventoryConfig())],
            objects={
                "wall": WallConfig(faction="vault"),
            },
        )

        cpp_config = convert_to_cpp_game_config(game_config)

        # Check that faction:vault tag is in the tag_id_map
        tag_names = list(cpp_config.tag_id_map.values())
        assert "faction:vault" in tag_names

    def test_multiple_objects_same_faction(self):
        """Test that multiple objects can share the same faction tag."""
        game_config = GameConfig(
            num_agents=1,
            resource_names=["gold"],
            factions=[FactionConfig(name="shared", inventory=InventoryConfig())],
            objects={
                "wall1": WallConfig(name="wall1", faction="shared"),
                "wall2": WallConfig(name="wall2", faction="shared"),
            },
        )

        cpp_config = convert_to_cpp_game_config(game_config)

        # Both walls should have the same faction tag
        # The tag should appear only once in the map
        tag_names = list(cpp_config.tag_id_map.values())
        assert tag_names.count("faction:shared") == 1
