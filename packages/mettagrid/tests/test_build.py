"""Tests for vibe-triggered build on move.

Build is triggered when:
1. Agent has a vibe that matches a BuildConfig on an object type
2. Agent moves successfully to a new location
3. Agent has sufficient resources to pay the build cost

The object is placed at the agent's previous location (where they moved from).
"""

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    BuildConfig,
    ChangeVibeActionConfig,
    GameConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    WallConfig,
)
from mettagrid.simulator import Simulation
from mettagrid.test_support.map_builders import ObjectNameMapBuilder


def get_agent_position(sim: Simulation, agent_id: int) -> tuple | None:
    """Get an agent's position as (row, col)."""
    grid_objects = sim.grid_objects()
    for obj in grid_objects.values():
        if obj.get("agent_id") == agent_id:
            return (obj["r"], obj["c"])
    return None


def get_agent_resources(sim: Simulation, agent_id: int) -> dict:
    """Get an agent's inventory resources by name."""
    return sim.agent(agent_id).inventory


def get_objects_at_position(sim: Simulation, row: int, col: int) -> list[dict]:
    """Get all objects at a specific position."""
    grid_objects = sim.grid_objects()
    return [obj for obj in grid_objects.values() if obj.get("r") == row and obj.get("c") == col]


def get_walls(sim: Simulation) -> list[dict]:
    """Get all wall objects in the simulation."""
    grid_objects = sim.grid_objects()
    return [obj for obj in grid_objects.values() if obj.get("type_name") == "wall"]


def count_buildable_walls(sim: Simulation, wall_type_name: str = "buildable_wall") -> int:
    """Count walls of a specific type."""
    grid_objects = sim.grid_objects()
    return sum(1 for obj in grid_objects.values() if obj.get("type_name") == wall_type_name)


class TestBuildOnMove:
    """Tests for build-on-move triggered by agent vibe."""

    def test_build_wall_on_move_with_matching_vibe(self):
        """When agent with builder vibe moves, a wall is built at the previous location."""
        map_data = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.red", "empty", "empty", "wall"],
            ["wall", "empty", "empty", "empty", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        config = GameConfig(
            max_steps=100,
            num_agents=1,
            obs=ObsConfig(width=5, height=5, num_tokens=50),
            resource_names=["energy", "carbon"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(),
                change_vibe=ChangeVibeActionConfig(number_of_vibes=20),
            ),
            objects={
                "wall": WallConfig(),
                "buildable_wall": WallConfig(
                    name="buildable_wall",
                    build=BuildConfig(
                        vibe="wall",  # Triggers when agent has "wall" vibe
                        cost={"carbon": 5},  # Costs 5 carbon to build
                    ),
                ),
            },
            agent=AgentConfig(
                inventory=InventoryConfig(
                    initial={"energy": 10, "carbon": 20},
                ),
            ),
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=map_data)
        sim = Simulation(mg_config, seed=42)

        # Verify initial state
        initial_pos = get_agent_position(sim, 0)
        assert initial_pos == (1, 1), f"Agent should start at (1,1), got {initial_pos}"

        initial_resources = get_agent_resources(sim, 0)
        assert initial_resources.get("carbon", 0) >= 5, "Agent should have at least 5 carbon"

        # Count initial buildable walls (should be 0 - we didn't place any)
        initial_wall_count = count_buildable_walls(sim)
        assert initial_wall_count == 0, "Should start with no buildable walls"

        # Change vibe to "wall" (the vibe that triggers building buildable_wall)
        sim.agent(0).set_action("change_vibe_wall")
        sim.step()

        # Move east - should trigger build at previous location
        sim.agent(0).set_action("move_east")
        sim.step()

        # Check agent moved
        new_pos = get_agent_position(sim, 0)
        assert new_pos == (1, 2), f"Agent should move to (1,2), got {new_pos}"

        # Check wall was built at previous location
        new_wall_count = count_buildable_walls(sim)
        assert new_wall_count == 1, f"Should have 1 buildable wall after move, got {new_wall_count}"

        # Check the wall is at the previous position
        objects_at_prev_pos = get_objects_at_position(sim, 1, 1)
        wall_at_prev = [obj for obj in objects_at_prev_pos if obj.get("type_name") == "buildable_wall"]
        assert len(wall_at_prev) == 1, "Wall should be at previous position (1,1)"

        # Check resources were consumed
        final_resources = get_agent_resources(sim, 0)
        expected_carbon = initial_resources.get("carbon", 0) - 5
        actual_carbon = final_resources.get("carbon", 0)
        assert actual_carbon == expected_carbon, (
            f"Agent should have 5 less carbon, had {initial_resources.get('carbon', 0)}, now has {actual_carbon}"
        )

    def test_no_build_without_matching_vibe(self):
        """Agent without builder vibe should not build when moving."""
        map_data = [
            ["wall", "wall", "wall", "wall"],
            ["wall", "agent.red", "empty", "wall"],
            ["wall", "wall", "wall", "wall"],
        ]

        config = GameConfig(
            max_steps=100,
            num_agents=1,
            obs=ObsConfig(width=5, height=5, num_tokens=50),
            resource_names=["energy", "carbon"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(),
                change_vibe=ChangeVibeActionConfig(number_of_vibes=20),
            ),
            objects={
                "wall": WallConfig(),
                "buildable_wall": WallConfig(
                    name="buildable_wall",
                    build=BuildConfig(
                        vibe="wall",
                        cost={"carbon": 5},
                    ),
                ),
            },
            agent=AgentConfig(
                inventory=InventoryConfig(
                    initial={"energy": 10, "carbon": 20},
                ),
            ),
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=map_data)
        sim = Simulation(mg_config, seed=42)

        # Move without changing vibe - should NOT build
        sim.agent(0).set_action("move_east")
        sim.step()

        # Verify agent moved
        new_pos = get_agent_position(sim, 0)
        assert new_pos == (1, 2), f"Agent should move to (1,2), got {new_pos}"

        # Verify no wall was built
        wall_count = count_buildable_walls(sim)
        assert wall_count == 0, "No wall should be built without matching vibe"

    def test_no_build_without_sufficient_resources(self):
        """Agent with matching vibe but insufficient resources should not build."""
        map_data = [
            ["wall", "wall", "wall", "wall"],
            ["wall", "agent.red", "empty", "wall"],
            ["wall", "wall", "wall", "wall"],
        ]

        config = GameConfig(
            max_steps=100,
            num_agents=1,
            obs=ObsConfig(width=5, height=5, num_tokens=50),
            resource_names=["energy", "carbon"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(),
                change_vibe=ChangeVibeActionConfig(number_of_vibes=20),
            ),
            objects={
                "wall": WallConfig(),
                "buildable_wall": WallConfig(
                    name="buildable_wall",
                    build=BuildConfig(
                        vibe="wall",
                        cost={"carbon": 50},  # Agent won't have enough
                    ),
                ),
            },
            agent=AgentConfig(
                inventory=InventoryConfig(
                    initial={"energy": 10, "carbon": 5},  # Only 5 carbon, need 50
                ),
            ),
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=map_data)
        sim = Simulation(mg_config, seed=42)

        # Change vibe to "wall"
        sim.agent(0).set_action("change_vibe_wall")
        sim.step()

        initial_carbon = get_agent_resources(sim, 0).get("carbon", 0)

        # Move - agent should move but NOT build (insufficient resources)
        sim.agent(0).set_action("move_east")
        sim.step()

        # Verify agent moved
        new_pos = get_agent_position(sim, 0)
        assert new_pos == (1, 2), f"Agent should move to (1,2), got {new_pos}"

        # Verify no wall was built
        wall_count = count_buildable_walls(sim)
        assert wall_count == 0, "No wall should be built without sufficient resources"

        # Verify resources were NOT consumed
        final_carbon = get_agent_resources(sim, 0).get("carbon", 0)
        assert final_carbon == initial_carbon, "Resources should not be consumed on failed build"

    def test_build_multiple_walls_in_sequence(self):
        """Agent can build multiple walls by moving multiple times."""
        map_data = [
            ["wall", "wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.red", "empty", "empty", "empty", "wall"],
            ["wall", "wall", "wall", "wall", "wall", "wall"],
        ]

        config = GameConfig(
            max_steps=100,
            num_agents=1,
            obs=ObsConfig(width=5, height=5, num_tokens=50),
            resource_names=["energy", "carbon"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(),
                change_vibe=ChangeVibeActionConfig(number_of_vibes=20),
            ),
            objects={
                "wall": WallConfig(),
                "buildable_wall": WallConfig(
                    name="buildable_wall",
                    build=BuildConfig(
                        vibe="wall",
                        cost={"carbon": 2},  # Low cost to allow multiple builds
                    ),
                ),
            },
            agent=AgentConfig(
                inventory=InventoryConfig(
                    initial={"energy": 10, "carbon": 20},
                ),
            ),
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=map_data)
        sim = Simulation(mg_config, seed=42)

        # Change vibe to "wall"
        sim.agent(0).set_action("change_vibe_wall")
        sim.step()

        # Move east 3 times, should build 3 walls
        for _ in range(3):
            sim.agent(0).set_action("move_east")
            sim.step()

        # Verify 3 walls were built
        wall_count = count_buildable_walls(sim)
        assert wall_count == 3, f"Should have 3 buildable walls, got {wall_count}"

        # Verify resources were consumed (3 * 2 = 6 carbon)
        final_resources = get_agent_resources(sim, 0)
        assert final_resources.get("carbon", 0) == 14, (
            f"Agent should have 14 carbon left (20 - 6), got {final_resources.get('carbon', 0)}"
        )

    def test_no_build_on_blocked_move(self):
        """When move fails (blocked), no build should occur."""
        map_data = [
            ["wall", "wall", "wall"],
            ["wall", "agent.red", "wall"],  # Agent blocked by wall
            ["wall", "wall", "wall"],
        ]

        config = GameConfig(
            max_steps=100,
            num_agents=1,
            obs=ObsConfig(width=5, height=5, num_tokens=50),
            resource_names=["energy", "carbon"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(),
                change_vibe=ChangeVibeActionConfig(number_of_vibes=20),
            ),
            objects={
                "wall": WallConfig(),
                "buildable_wall": WallConfig(
                    name="buildable_wall",
                    build=BuildConfig(
                        vibe="wall",
                        cost={"carbon": 5},
                    ),
                ),
            },
            agent=AgentConfig(
                inventory=InventoryConfig(
                    initial={"energy": 10, "carbon": 20},
                ),
            ),
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=map_data)
        sim = Simulation(mg_config, seed=42)

        # Change vibe to "wall"
        sim.agent(0).set_action("change_vibe_wall")
        sim.step()

        initial_carbon = get_agent_resources(sim, 0).get("carbon", 0)
        initial_pos = get_agent_position(sim, 0)

        # Try to move east (blocked by wall)
        sim.agent(0).set_action("move_east")
        sim.step()

        # Verify agent didn't move
        new_pos = get_agent_position(sim, 0)
        assert new_pos == initial_pos, "Agent should not move when blocked"

        # Verify no extra wall was built
        buildable_wall_count = count_buildable_walls(sim)
        assert buildable_wall_count == 0, "No wall should be built on blocked move"

        # Verify resources were NOT consumed
        final_carbon = get_agent_resources(sim, 0).get("carbon", 0)
        assert final_carbon == initial_carbon, "Resources should not be consumed on failed move"


class TestBuildConfig:
    """Tests for BuildConfig validation and configuration."""

    def test_build_config_on_wall(self):
        """Verify BuildConfig can be attached to WallConfig."""
        wall_config = WallConfig(
            name="buildable_wall",
            build=BuildConfig(
                vibe="wall",
                cost={"carbon": 5, "energy": 2},
            ),
        )

        assert wall_config.build is not None
        assert wall_config.build.vibe == "wall"
        assert wall_config.build.cost == {"carbon": 5, "energy": 2}

    def test_wall_without_build_config(self):
        """Standard wall should have no build config."""
        wall_config = WallConfig()
        assert wall_config.build is None

    def test_build_config_empty_cost(self):
        """BuildConfig with empty cost should be valid (free build)."""
        build_config = BuildConfig(vibe="default", cost={})
        assert build_config.cost == {}

    def test_build_config_serialization(self):
        """BuildConfig should serialize correctly."""
        config = GameConfig(
            max_steps=10,
            num_agents=1,
            resource_names=["carbon"],
            objects={
                "wall": WallConfig(),
                "buildable_wall": WallConfig(
                    name="buildable_wall",
                    build=BuildConfig(vibe="wall", cost={"carbon": 5}),
                ),
            },
        )

        # Serialize and check the build config is preserved
        dumped = config.model_dump()
        assert dumped["objects"]["buildable_wall"]["build"] is not None
        assert dumped["objects"]["buildable_wall"]["build"]["vibe"] == "wall"
        assert dumped["objects"]["buildable_wall"]["build"]["cost"] == {"carbon": 5}
