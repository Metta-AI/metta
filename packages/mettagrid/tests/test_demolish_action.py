"""Tests for the demolish action.

Demolish is triggered when an agent with an attack vibe walks into a building
that has a demolish config. If the agent has the required resources, the building
is destroyed and scrap resources are returned to the agent.
"""

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AttackActionConfig,
    ChangeVibeActionConfig,
    DemolishConfig,
    GameConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    WallConfig,
)
from mettagrid.config.vibes import Vibe
from mettagrid.simulator import Simulation
from mettagrid.test_support.map_builders import ObjectNameMapBuilder

# Define test vibes
TEST_VIBES = [
    Vibe("😐", "default", category="emotion"),
    Vibe("⚔️", "weapon", category="combat"),
]


def create_demolish_test_sim(
    demolish_config: DemolishConfig | None = None,
    initial_inventory: dict[str, int] | None = None,
) -> Simulation:
    """Create a simulation for testing demolish action."""
    # Map with agent, barrier (demolishable), and walls
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "agent.agent", "barrier", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    vibe_names = [v.name for v in TEST_VIBES]

    game_config = GameConfig(
        max_steps=50,
        num_agents=1,
        obs=ObsConfig(width=3, height=3, num_tokens=100),
        resource_names=["energy", "carbon", "silicon"],
        vibe_names=vibe_names,
        actions=ActionsConfig(
            noop=NoopActionConfig(),
            move=MoveActionConfig(enabled=True),
            change_vibe=ChangeVibeActionConfig(enabled=True, vibes=TEST_VIBES, number_of_vibes=len(TEST_VIBES)),
            attack=AttackActionConfig(
                enabled=True,
                vibes=["weapon"],  # Attack triggered when vibing weapon
            ),
        ),
        agent=AgentConfig(
            rewards=AgentRewards(),
            initial_inventory=initial_inventory or {"energy": 100, "carbon": 50, "silicon": 50},
        ),
        objects={
            "wall": WallConfig(),  # Non-demolishable
            "barrier": WallConfig(
                name="barrier",
                render_symbol="🪨",
                demolish=demolish_config,
            ),
        },
    )

    cfg = MettaGridConfig(game=game_config)
    cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

    return Simulation(cfg, seed=42)


def count_barriers(sim: Simulation) -> int:
    """Count barrier objects in the simulation."""
    objects = sim.grid_objects()
    return sum(1 for obj in objects.values() if obj.get("type_name") == "barrier")


def get_agent(sim: Simulation) -> dict:
    """Get the agent object from the simulation."""
    objects = sim.grid_objects()
    agents = [obj for obj in objects.values() if "agent_id" in obj]
    return agents[0] if agents else {}


class TestDemolishAction:
    """Test demolish action triggered by attack vibe on buildings."""

    def test_demolish_barrier_with_sufficient_resources(self):
        """Test that walking into a barrier with weapon vibe demolishes it."""
        sim = create_demolish_test_sim(
            demolish_config=DemolishConfig(
                cost={"energy": 20},
                scrap={"silicon": 10},
            ),
            initial_inventory={"energy": 100, "carbon": 50, "silicon": 0},
        )

        initial_barriers = count_barriers(sim)
        assert initial_barriers == 1, f"Should start with 1 barrier, got {initial_barriers}"

        # Get initial inventory (inventory is a dict keyed by resource ID)
        agent = get_agent(sim)
        energy_idx = sim.resource_names.index("energy")
        silicon_idx = sim.resource_names.index("silicon")
        energy_before = agent["inventory"].get(energy_idx, 0)
        silicon_before = agent["inventory"].get(silicon_idx, 0)

        # Agent changes vibe to "weapon"
        sim.agent(0).set_action("change_vibe_weapon")
        sim.step()

        # Verify vibe is set
        agent = get_agent(sim)
        assert agent["vibe"] == 1, f"Agent should have weapon vibe (id=1), got {agent['vibe']}"

        # Agent walks east into barrier - should demolish it
        sim.agent(0).set_action("move_east")
        sim.step()

        # Check that barrier was demolished
        final_barriers = count_barriers(sim)
        assert final_barriers == 0, f"Barrier should be demolished, but found {final_barriers}"

        # Check that resources were updated
        agent_after = get_agent(sim)
        energy_after = agent_after["inventory"].get(energy_idx, 0)
        silicon_after = agent_after["inventory"].get(silicon_idx, 0)

        assert energy_after == energy_before - 20, (
            f"Energy should be reduced by 20. Was {energy_before}, now {energy_after}"
        )
        assert silicon_after == silicon_before + 10, (
            f"Silicon should increase by 10 (scrap). Was {silicon_before}, now {silicon_after}"
        )

    def test_no_demolish_without_attack_vibe(self):
        """Test that walking into barrier without weapon vibe does NOT demolish it."""
        sim = create_demolish_test_sim(
            demolish_config=DemolishConfig(
                cost={"energy": 20},
                scrap={"silicon": 10},
            ),
            initial_inventory={"energy": 100, "carbon": 50, "silicon": 0},
        )

        initial_barriers = count_barriers(sim)

        # Agent keeps default vibe and tries to walk east into barrier
        sim.agent(0).set_action("move_east")
        sim.step()

        # Barrier should still exist (move blocked or no demolish)
        final_barriers = count_barriers(sim)
        assert final_barriers == initial_barriers, (
            f"Barrier should NOT be demolished without weapon vibe. Had {initial_barriers}, now {final_barriers}"
        )

    def test_no_demolish_insufficient_resources(self):
        """Test that demolish fails when agent doesn't have enough resources."""
        sim = create_demolish_test_sim(
            demolish_config=DemolishConfig(
                cost={"energy": 200},  # More than agent has
                scrap={"silicon": 10},
            ),
            initial_inventory={"energy": 100, "carbon": 50, "silicon": 0},
        )

        initial_barriers = count_barriers(sim)

        # Agent changes vibe to "weapon"
        sim.agent(0).set_action("change_vibe_weapon")
        sim.step()

        # Agent tries to walk east into barrier - should fail (insufficient energy)
        sim.agent(0).set_action("move_east")
        sim.step()

        # Barrier should still exist
        final_barriers = count_barriers(sim)
        assert final_barriers == initial_barriers, (
            f"Barrier should NOT be demolished (insufficient resources). Had {initial_barriers}, now {final_barriers}"
        )

        # Check stats for failed demolish
        stats = sim.episode_stats
        failed_count = stats["agent"][0].get("action.attack.red.demolish.failed.insufficient_resources", 0)
        assert failed_count > 0, "Should have recorded insufficient_resources failure"

    def test_no_demolish_without_demolish_config(self):
        """Test that buildings without demolish config cannot be demolished."""
        sim = create_demolish_test_sim(
            demolish_config=None,  # No demolish config
            initial_inventory={"energy": 100, "carbon": 50, "silicon": 0},
        )

        initial_barriers = count_barriers(sim)

        # Agent changes vibe to "weapon"
        sim.agent(0).set_action("change_vibe_weapon")
        sim.step()

        # Agent tries to walk east into barrier - should not demolish (no config)
        sim.agent(0).set_action("move_east")
        sim.step()

        # Barrier should still exist
        final_barriers = count_barriers(sim)
        assert final_barriers == initial_barriers, (
            f"Barrier should NOT be demolished (no demolish config). Had {initial_barriers}, now {final_barriers}"
        )

    def test_demolish_stats_tracked(self):
        """Test that demolish statistics are tracked."""
        sim = create_demolish_test_sim(
            demolish_config=DemolishConfig(
                cost={"energy": 10},
                scrap={"silicon": 5},
            ),
            initial_inventory={"energy": 100, "carbon": 50, "silicon": 0},
        )

        # Agent changes vibe to "weapon" and demolishes barrier
        sim.agent(0).set_action("change_vibe_weapon")
        sim.step()
        sim.agent(0).set_action("move_east")
        sim.step()

        # Check stats - agent level with full action prefix
        stats = sim.episode_stats
        demolish_count = stats["agent"][0].get("action.attack.red.demolish.barrier", 0)
        assert demolish_count == 1, f"action.attack.red.demolish.barrier should be 1, got {demolish_count}"

        # Check simple demolish stat (also on agent)
        simple_demolish = stats["agent"][0].get("demolish.barrier", 0)
        assert simple_demolish == 1, f"demolish.barrier should be 1, got {simple_demolish}"

    def test_demolish_zero_cost(self):
        """Test that demolish with zero cost still works."""
        sim = create_demolish_test_sim(
            demolish_config=DemolishConfig(
                cost={},  # Free to demolish
                scrap={"silicon": 15},
            ),
            initial_inventory={"energy": 0, "carbon": 0, "silicon": 0},  # No resources
        )

        # Agent changes vibe to "weapon"
        sim.agent(0).set_action("change_vibe_weapon")
        sim.step()

        # Agent walks east into barrier - should demolish for free
        sim.agent(0).set_action("move_east")
        sim.step()

        # Barrier should be demolished
        final_barriers = count_barriers(sim)
        assert final_barriers == 0, f"Barrier should be demolished (free), but found {final_barriers}"

        # Agent should have received scrap
        agent = get_agent(sim)
        silicon_idx = sim.resource_names.index("silicon")
        silicon = agent["inventory"].get(silicon_idx, 0)
        assert silicon == 15, f"Agent should have 15 silicon (scrap), got {silicon}"


class TestDemolishConfig:
    """Test DemolishConfig validation and behavior."""

    def test_demolish_config_defaults(self):
        """Test that DemolishConfig has sensible defaults."""
        config = DemolishConfig()
        assert config.cost == {}
        assert config.scrap == {}

    def test_demolish_config_with_values(self):
        """Test DemolishConfig with custom values."""
        config = DemolishConfig(
            cost={"energy": 10, "carbon": 5},
            scrap={"silicon": 20},
        )
        assert config.cost == {"energy": 10, "carbon": 5}
        assert config.scrap == {"silicon": 20}
