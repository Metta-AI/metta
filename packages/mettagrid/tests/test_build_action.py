"""Tests for the build action.

Build is triggered when an agent with a build vibe successfully moves to an empty location.
When triggered, a wall (or other object) is placed at the agent's previous position,
and the configured resource cost is deducted from the agent's inventory.
"""

import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    BuildConfig,
    ChangeVibeActionConfig,
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
    Vibe("⬛", "wall", category="building"),
]


def create_build_test_sim(
    build_config: BuildConfig | None = None,
    initial_inventory: dict[str, int] | None = None,
) -> Simulation:
    """Create a simulation for testing build action."""
    # Map with agent and empty space to move into
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "agent.agent", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    vibe_names = [v.name for v in TEST_VIBES]

    # Create wall config with optional build config
    wall_config = WallConfig(build=build_config)

    game_config = GameConfig(
        max_steps=50,
        num_agents=1,
        obs=ObsConfig(width=3, height=3, num_tokens=100),
        resource_names=["energy", "carbon"],
        vibe_names=vibe_names,
        actions=ActionsConfig(
            noop=NoopActionConfig(),
            move=MoveActionConfig(enabled=True),
            change_vibe=ChangeVibeActionConfig(enabled=True, vibes=TEST_VIBES, number_of_vibes=len(TEST_VIBES)),
        ),
        agent=AgentConfig(
            rewards=AgentRewards(),
            initial_inventory=initial_inventory or {"energy": 100, "carbon": 50},
        ),
        objects={
            "wall": wall_config,
        },
    )

    cfg = MettaGridConfig(game=game_config)
    cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

    return Simulation(cfg, seed=42)


def count_walls(sim: Simulation) -> int:
    """Count wall objects in the simulation."""
    objects = sim.grid_objects()
    return sum(1 for obj in objects.values() if obj.get("type_name") == "wall")


def get_agent(sim: Simulation) -> dict:
    """Get the agent object from the simulation."""
    objects = sim.grid_objects()
    agents = [obj for obj in objects.values() if "agent_id" in obj]
    return agents[0] if agents else {}


class TestBuildAction:
    """Test build action triggered by vibes on move."""

    def test_build_triggered_by_wall_vibe(self):
        """Test that moving to empty space with wall vibe builds a wall at previous position."""
        sim = create_build_test_sim(
            build_config=BuildConfig(vibe="wall", cost={"energy": 10, "carbon": 5}),
            initial_inventory={"energy": 100, "carbon": 50},
        )

        initial_walls = count_walls(sim)

        # Agent changes vibe to "wall"
        sim.agent(0).set_action("change_vibe_wall")
        sim.step()

        # Verify vibe is set (wall should be vibe id 1)
        agent = get_agent(sim)
        assert agent["vibe"] == 1, f"Agent should have wall vibe (id=1), got {agent['vibe']}"

        # Get inventory before build
        energy_idx = sim.resource_names.index("energy")
        carbon_idx = sim.resource_names.index("carbon")
        energy_before = agent["inventory"][energy_idx]
        carbon_before = agent["inventory"][carbon_idx]

        # Agent moves west to empty space - should trigger build at previous position
        sim.agent(0).set_action("move_west")
        sim.step()

        # Check that a wall was built
        final_walls = count_walls(sim)
        assert final_walls == initial_walls + 1, (
            f"Should have built 1 wall. Had {initial_walls}, now have {final_walls}"
        )

        # Check that resources were deducted
        agent_after = get_agent(sim)
        energy_after = agent_after["inventory"][energy_idx]
        carbon_after = agent_after["inventory"][carbon_idx]

        assert energy_after == energy_before - 10, (
            f"Energy should be reduced by 10. Was {energy_before}, now {energy_after}"
        )
        assert carbon_after == carbon_before - 5, (
            f"Carbon should be reduced by 5. Was {carbon_before}, now {carbon_after}"
        )

    def test_no_build_without_wall_vibe(self):
        """Test that moving to empty space without wall vibe does NOT build a wall."""
        sim = create_build_test_sim(
            build_config=BuildConfig(vibe="wall", cost={"energy": 10, "carbon": 5}),
            initial_inventory={"energy": 100, "carbon": 50},
        )

        initial_walls = count_walls(sim)

        # Agent keeps default vibe (not wall)
        # Get inventory before move
        agent = get_agent(sim)
        carbon_idx = sim.resource_names.index("carbon")
        carbon_before = agent["inventory"][carbon_idx]

        # Agent moves west to empty space - should NOT trigger build
        sim.agent(0).set_action("move_west")
        sim.step()

        # Check that no wall was built
        final_walls = count_walls(sim)
        assert final_walls == initial_walls, (
            f"Should NOT have built any walls. Had {initial_walls}, now have {final_walls}"
        )

        # Check that resources are unchanged (except move cost if configured)
        agent_after = get_agent(sim)
        # Note: move action may consume resources too, so we only check carbon which is not used by move
        carbon_after = agent_after["inventory"][carbon_idx]
        assert carbon_after == carbon_before, f"Carbon should be unchanged. Was {carbon_before}, now {carbon_after}"

    def test_no_build_insufficient_resources(self):
        """Test that build fails when agent doesn't have enough resources."""
        sim = create_build_test_sim(
            build_config=BuildConfig(vibe="wall", cost={"energy": 10, "carbon": 5}),
            initial_inventory={"energy": 100, "carbon": 2},  # Not enough carbon
        )

        initial_walls = count_walls(sim)

        # Agent changes vibe to "wall"
        sim.agent(0).set_action("change_vibe_wall")
        sim.step()

        # Agent moves west - should move but NOT build (insufficient carbon)
        sim.agent(0).set_action("move_west")
        sim.step()

        # Check that no wall was built
        final_walls = count_walls(sim)
        assert final_walls == initial_walls, (
            f"Should NOT have built any walls (insufficient resources). Had {initial_walls}, now have {final_walls}"
        )

        # Check stats for failed build
        stats = sim.episode_stats
        failed_count = stats["agent"][0].get("action.build.red.failed.insufficient_resources", 0)
        assert failed_count > 0, "Should have recorded insufficient_resources failure"

    def test_multiple_builds_in_sequence(self):
        """Test building multiple walls in sequence."""
        # Create a larger map for multiple builds
        game_map = [
            ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
            ["wall", "empty", "empty", "agent.agent", "empty", "empty", "wall"],
            ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = GameConfig(
            max_steps=50,
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=100),
            resource_names=["energy", "carbon"],
            vibe_names=[v.name for v in TEST_VIBES],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(enabled=True),
                change_vibe=ChangeVibeActionConfig(enabled=True, vibes=TEST_VIBES, number_of_vibes=len(TEST_VIBES)),
            ),
            agent=AgentConfig(
                rewards=AgentRewards(),
                initial_inventory={"energy": 100, "carbon": 50},
            ),
            objects={
                "wall": WallConfig(build=BuildConfig(vibe="wall", cost={"energy": 10, "carbon": 5})),
            },
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
        sim = Simulation(cfg, seed=42)

        initial_walls = count_walls(sim)

        # Change to wall vibe
        sim.agent(0).set_action("change_vibe_wall")
        sim.step()

        # Build first wall by moving west
        sim.agent(0).set_action("move_west")
        sim.step()

        # Build second wall by moving west again
        sim.agent(0).set_action("move_west")
        sim.step()

        # Should have built 2 walls
        final_walls = count_walls(sim)
        assert final_walls == initial_walls + 2, (
            f"Should have built 2 walls. Had {initial_walls}, now have {final_walls}"
        )

        # Check stats
        stats = sim.episode_stats
        build_count = stats["agent"][0].get("action.build.red.count", 0)
        assert build_count == 2, f"Build count should be 2, got {build_count}"

    def test_build_move_blocked_no_wall_created(self):
        """Test that no wall is created when move is blocked (move into wall)."""
        sim = create_build_test_sim(
            build_config=BuildConfig(vibe="wall", cost={"energy": 10, "carbon": 5}),
            initial_inventory={"energy": 100, "carbon": 50},
        )

        initial_walls = count_walls(sim)

        # Change to wall vibe
        sim.agent(0).set_action("change_vibe_wall")
        sim.step()

        # Get resources before failed move
        agent = get_agent(sim)
        carbon_idx = sim.resource_names.index("carbon")
        carbon_before = agent["inventory"][carbon_idx]

        # Try to move north into wall - should fail
        sim.agent(0).set_action("move_north")
        sim.step()

        # No wall should be built (move failed)
        final_walls = count_walls(sim)
        assert final_walls == initial_walls, (
            f"Should NOT have built any walls (move blocked). Had {initial_walls}, now have {final_walls}"
        )

        # Resources should be unchanged
        agent_after = get_agent(sim)
        carbon_after = agent_after["inventory"][carbon_idx]
        assert carbon_after == carbon_before, (
            f"Carbon should be unchanged when move fails. Was {carbon_before}, now {carbon_after}"
        )


class TestBuildConfigValidation:
    """Test configuration validation for build action."""

    def test_invalid_vibe_in_build_config_raises_error(self):
        """Test that using an invalid vibe name in object build config raises an error."""
        with pytest.raises(ValueError, match="Unknown vibe name"):
            game_config = GameConfig(
                max_steps=50,
                num_agents=1,
                resource_names=["energy", "carbon"],
                vibe_names=["default", "wall"],
                actions=ActionsConfig(
                    noop=NoopActionConfig(),
                ),
                objects={
                    "wall": WallConfig(
                        build=BuildConfig(vibe="nonexistent_vibe", cost={"energy": 10}),
                    ),
                },
            )
            cfg = MettaGridConfig(game=game_config)
            cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=[["agent.agent", "empty"]])
            Simulation(cfg)
