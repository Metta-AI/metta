from cogames.cogs_vs_clips.missions import HarvestMission, make_game
from cogames.cogs_vs_clips.variants import InventoryHeartTuneVariant, NeutralFacedVariant
from mettagrid.config.mettagrid_config import AssemblerConfig, MettaGridConfig


def test_make_cogs_vs_clips_scenario():
    """Test that make_cogs_vs_clips_scenario creates a valid configuration."""
    # Create the scenario
    config = make_game()

    # Verify it returns a MettaGridConfig
    assert isinstance(config, MettaGridConfig)

    # # Check game configuration
    # assert config.game is not None
    # assert config.game.num_agents == 2

    # # Check actions configuration
    # assert config.game.actions is not None
    # assert hasattr(config.game.actions, "move")
    # assert hasattr(config.game.actions, "noop")
    # assert hasattr(config.game.actions, "rotate")

    # # Check objects configuration
    # assert config.game.objects is not None
    # assert "wall" in config.game.objects
    # assert config.game.objects["wall"].type_id == 1

    # # Check map builder configuration
    # assert config.game.map_builder is not None
    # assert isinstance(config.game.map_builder, RandomMapBuilder.Config)
    # assert config.game.map_builder.width == 10
    # assert config.game.map_builder.height == 10
    # assert config.game.map_builder.agents == 2
    # assert config.game.map_builder.seed == 42

    # # Check agent configuration
    # assert config.game.agent is not None
    # assert config.game.agent.default_resource_limit == 10
    # assert config.game.agent.resource_limits == {"heart": 10}
    # assert config.game.agent.rewards is not None
    # assert config.game.agent.rewards.inventory == {}


def test_neutral_faced_variant_neutralizes_recipes():
    mission = HarvestMission
    assert mission.site is not None
    mission = mission.with_variants([NeutralFacedVariant()])
    env = mission.make_env()

    change_vibe = env.game.actions.change_vibe
    assert change_vibe.enabled is False
    assert change_vibe.number_of_vibes == 1

    for obj in env.game.objects.values():
        if not isinstance(obj, AssemblerConfig):
            continue
        assert len(obj.protocols) == 1
        protocol = obj.protocols[0]
        assert protocol.vibes == ["default"]


def test_inventory_heart_tune_caps_initial_inventory_to_limits():
    mission = HarvestMission.with_variants([InventoryHeartTuneVariant(hearts=20)])
    env = mission.make_env()
    agent = env.game.agent

    energy_limit = int(agent.resource_limits["energy"])
    assert agent.initial_inventory["energy"] == energy_limit
