"""Tests for grid_objects() method properties and ignore_types parameter."""

import numpy as np
import pytest

from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AssemblerConfig,
    ChestConfig,
    GameConfig,
    MettaGridConfig,
    ProtocolConfig,
    WallConfig,
)
from mettagrid.core import BoundingBox, MettaGridCore
from mettagrid.map_builder.random import RandomMapBuilder


@pytest.fixture
def env_with_assembler():
    """Create environment with an assembler to test assembler properties."""
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=2,
            max_steps=100,
            obs_width=5,
            obs_height=5,
            num_observation_tokens=100,
            resource_names=["iron", "steel"],
            actions=ActionsConfig(
                noop=ActionConfig(),
                move=ActionConfig(),
            ),
            objects={
                "wall": WallConfig(),
                "assembler": AssemblerConfig(
                    recipes=[
                        ([], ProtocolConfig(input_resources={"iron": 10}, output_resources={"steel": 5}, cooldown=20))
                    ],
                    max_uses=10,
                    allow_partial_usage=True,
                    exhaustion=0.1,
                ),
            },
            map_builder=RandomMapBuilder.Config(
                width=10,
                height=10,
                agents=2,
                objects={"assembler": 1},  # Add one assembler
                seed=42,
            ),
        )
    )
    return MettaGridCore(config)


@pytest.fixture
def env_with_chest():
    """Create environment with a chest to test chest properties."""
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            obs_width=5,
            obs_height=5,
            num_observation_tokens=100,
            resource_names=["gold", "silver"],
            actions=ActionsConfig(
                noop=ActionConfig(),
                move=ActionConfig(),
            ),
            objects={
                "wall": WallConfig(),
                "chest": ChestConfig(
                    resource_type="gold",
                    position_deltas=[("NW", 1), ("N", 1), ("NE", 1), ("SW", -1), ("S", -1), ("SE", -1)],
                ),
            },
            map_builder=RandomMapBuilder.Config(
                width=10,
                height=10,
                agents=1,
                objects={"chest": 1},  # Add one chest
                seed=42,
            ),
        )
    )
    return MettaGridCore(config)


@pytest.fixture
def env_with_walls():
    """Create environment with walls to test ignore_types."""
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=2,
            max_steps=10,
            obs_width=5,
            obs_height=5,
            num_observation_tokens=50,
            actions=ActionsConfig(
                noop=ActionConfig(),
                move=ActionConfig(),
            ),
            objects={
                "wall": WallConfig(),
            },
            map_builder=RandomMapBuilder.Config(
                width=15,
                height=15,
                agents=2,
                objects={"wall": 20},  # Add 20 random walls
                border_width=1,  # Add border walls
                seed=123,
            ),
        )
    )
    return MettaGridCore(config)


def _find_object_primary(env: MettaGridCore, type_name: str) -> tuple[int, tuple[int, int, int]]:
    env_objects = env.grid_objects()
    for object_id, obj in env_objects.items():
        if obj.get("type_name") == type_name:
            locations = obj["locations"]
            if locations:
                c, r, layer = locations[0]
                return object_id, (r, c, layer)
    raise AssertionError(f"Expected to find object of type '{type_name}'")


class TestIgnoreTypes:
    """Test ignore_types parameter for filtering objects."""

    def test_ignore_types_walls(self, env_with_walls):
        """Test that ignore_types=['wall'] correctly filters out walls."""
        env_with_walls.reset()

        # Get all objects
        all_objects = env_with_walls.grid_objects()
        print(f"All objects count: {len(all_objects)}")

        # Get objects without walls
        no_walls = env_with_walls.grid_objects(ignore_types=["wall"])
        print(f"Objects without walls: {len(no_walls)}")

        # Count walls manually
        wall_count = sum(1 for obj in all_objects.values() if obj.get("type_name") == "wall")
        agent_count = sum(1 for obj in all_objects.values() if obj.get("type_name") == "agent")

        print(f"Wall count: {wall_count}")
        print(f"Agent count: {agent_count}")

        # Verify filtering worked
        assert len(all_objects) - len(no_walls) == wall_count, "Filtered count should match wall count"

        # Verify no walls remain in filtered result
        remaining_types = set(obj.get("type_name") for obj in no_walls.values())
        assert "wall" not in remaining_types, "No walls should remain after filtering"

        # Verify agents are still there
        agent_count_filtered = sum(1 for obj in no_walls.values() if obj.get("type_name") == "agent")
        assert agent_count_filtered == agent_count, "All agents should still be present"

    def test_ignore_multiple_types(self, env_with_walls):
        """Test ignoring multiple object types."""
        env_with_walls.reset()

        all_objects = env_with_walls.grid_objects()

        # Filter out both walls and agents
        no_walls_or_agents = env_with_walls.grid_objects(ignore_types=["wall", "agent"])

        # Count types
        wall_count = sum(1 for obj in all_objects.values() if obj.get("type_name") == "wall")
        agent_count = sum(1 for obj in all_objects.values() if obj.get("type_name") == "agent")

        # Should have filtered out all objects in this simple environment
        expected_remaining = len(all_objects) - wall_count - agent_count
        assert len(no_walls_or_agents) == expected_remaining

    def test_ignore_with_bounding_box(self, env_with_walls):
        """Test that ignore_types works with bounding box filtering."""
        env_with_walls.reset()

        bbox = BoundingBox(min_row=0, max_row=5, min_col=0, max_col=5)

        # Get objects in bbox
        bbox_objects = env_with_walls.grid_objects(bbox=bbox)

        # Get objects in bbox without walls
        bbox_no_walls = env_with_walls.grid_objects(bbox=bbox, ignore_types=["wall"])

        # Count walls in bbox
        wall_count = sum(1 for obj in bbox_objects.values() if obj.get("type_name") == "wall")

        assert len(bbox_objects) - len(bbox_no_walls) == wall_count


class TestAssemblerProperties:
    """Test assembler-specific properties in grid_objects."""

    def test_assembler_basic_properties(self, env_with_assembler):
        """Test that basic assembler properties are exposed."""
        env_with_assembler.reset()

        objects = env_with_assembler.grid_objects()

        # Find an assembler
        assembler = next((obj for obj in objects.values() if obj.get("type_name") == "assembler"), None)

        if assembler:
            # Check basic properties exist
            assert "cooldown_remaining" in assembler
            assert "cooldown_duration" in assembler
            assert "cooldown_progress" in assembler
            assert "is_clipped" in assembler
            assert "is_clip_immune" in assembler
            assert "uses_count" in assembler
            assert "max_uses" in assembler
            assert "allow_partial_usage" in assembler
            assert "exhaustion" in assembler
            assert "cooldown_multiplier" in assembler

            # Check initial values
            assert assembler["cooldown_remaining"] == 0, "Should start with no cooldown"
            assert assembler["is_clipped"] is False, "Should not be clipped initially"
            assert assembler["uses_count"] == 0, "Should start with zero uses"
            assert assembler["max_uses"] == 10, "Max uses should match config"
            assert assembler["allow_partial_usage"] is True, "Should match config"
            assert assembler["exhaustion"] == pytest.approx(0.1), "Exhaustion should match config"
            assert assembler["cooldown_multiplier"] == pytest.approx(1.0), "Should start at 1.0"

    def test_assembler_current_recipe(self, env_with_assembler):
        """Test that current recipe is exposed."""
        env_with_assembler.reset()

        objects = env_with_assembler.grid_objects()

        # Find an assembler
        assembler = next((obj for obj in objects.values() if obj.get("type_name") == "assembler"), None)

        if assembler:
            # Recipe properties may or may not exist depending on whether agents are nearby
            if "current_recipe_inputs" in assembler:
                assert isinstance(assembler["current_recipe_inputs"], dict)

            if "current_recipe_outputs" in assembler:
                assert isinstance(assembler["current_recipe_outputs"], dict)

            # If recipe exists, verify structure
            if "current_recipe_inputs" in assembler:
                assert isinstance(assembler["current_recipe_inputs"], dict)
                assert "current_recipe_outputs" in assembler
                assert isinstance(assembler["current_recipe_outputs"], dict)
                assert "current_recipe_cooldown" in assembler
                assert isinstance(assembler["current_recipe_cooldown"], int)

    def test_assembler_all_recipes(self, env_with_assembler):
        """Test that all recipes are exposed."""
        env_with_assembler.reset()

        objects = env_with_assembler.grid_objects()

        # Find an assembler
        assembler = next((obj for obj in objects.values() if obj.get("type_name") == "assembler"), None)

        if assembler:
            # Check that recipes list exists
            assert "recipes" in assembler
            assert isinstance(assembler["recipes"], list)

            # Check that at least one recipe exists (we defined one in the fixture)
            assert len(assembler["recipes"]) > 0

            # Verify structure of first recipe
            recipe = assembler["recipes"][0]
            assert "inputs" in recipe
            assert "outputs" in recipe
            assert "cooldown" in recipe
            assert isinstance(recipe["inputs"], dict)
            assert isinstance(recipe["outputs"], dict)
            assert isinstance(recipe["cooldown"], int)

    @staticmethod
    def _add_linear_locations(
        env: MettaGridCore,
        object_id: int,
        r: int,
        c: int,
        layer: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        directions = ((0, 1), (1, 0), (0, -1), (-1, 0))
        for dr, dc in directions:
            first = (r + dr, c + dc)
            ok_first = env.add_object_location(object_id, first[0], first[1], layer)
            if not ok_first:
                continue
            second = (r + 2 * dr, c + 2 * dc)
            ok_second = env.add_object_location(object_id, second[0], second[1], layer)
            if ok_second:
                return first, second
            removed = env.remove_object_location(object_id, first[0], first[1], layer)
            assert removed, "Failed to remove temporary cell after unsuccessful attempt"
        raise AssertionError("Could not add two linear locations for assembler test")

    def test_assembler_multi_cell_add_remove_and_bbox(self, env_with_assembler):
        env_with_assembler.reset()

        assembler_id, (r, c, layer) = _find_object_primary(env_with_assembler, "assembler")

        first_cell, second_cell = self._add_linear_locations(env_with_assembler, assembler_id, r, c, layer)

        objects = env_with_assembler.grid_objects()
        assembler = objects[assembler_id]
        locations = set((rr, cc, ll) for cc, rr, ll in assembler.get("locations", []))
        assert (r, c, layer) in locations
        assert (first_cell[0], first_cell[1], layer) in locations
        assert (second_cell[0], second_cell[1], layer) in locations

        removed_middle = env_with_assembler.remove_object_location(assembler_id, first_cell[0], first_cell[1], layer)
        assert removed_middle, "Removing middle cell should now succeed"

        removed_far = env_with_assembler.remove_object_location(assembler_id, second_cell[0], second_cell[1], layer)
        assert removed_far, "Removing outer cell should succeed"

        bbox_far = BoundingBox(
            min_row=second_cell[0],
            max_row=second_cell[0] + 1,
            min_col=second_cell[1],
            max_col=second_cell[1] + 1,
        )
        objs_far = env_with_assembler.grid_objects(bbox=bbox_far)
        assert assembler_id not in objs_far, (
            "Assembler should be excluded when only far cell area is queried after removal"
        )

        bbox_adj = BoundingBox(
            min_row=min(r, first_cell[0]),
            max_row=max(r, first_cell[0]) + 1,
            min_col=min(c, first_cell[1]),
            max_col=max(c, first_cell[1]) + 1,
        )
        objs_adj = env_with_assembler.grid_objects(bbox=bbox_adj)
        assert assembler_id in objs_adj, "Assembler should be included when any occupied cell intersects bbox"

    def test_assembler_add_object_cell_rejects_diagonal_only_connection(self, env_with_assembler):
        env_with_assembler.reset()

        assembler_id, (r, c, layer) = _find_object_primary(env_with_assembler, "assembler")
        height = env_with_assembler.map_height
        width = env_with_assembler.map_width

        occupied = set()
        objects = env_with_assembler.grid_objects()
        for obj in objects.values():
            for c_loc, r_loc, layer_loc in obj.get("locations", []):
                occupied.add((r_loc, c_loc, layer_loc))

        diagonal_target = None
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= height:
                continue
            if nc < 0 or nc >= width:
                continue
            if (nr, nc, layer) in occupied:
                continue
            diagonal_target = (nr, nc)
            break
        assert diagonal_target is not None, "Expected a free diagonal cell near assembler for test setup"

        ok = env_with_assembler.add_object_location(assembler_id, diagonal_target[0], diagonal_target[1], layer)
        assert ok, "Diagonal connections should now be allowed"

    def test_assembler_add_object_cell_rejects_other_layer(self, env_with_assembler):
        env_with_assembler.reset()

        assembler_id, (r, c, layer) = _find_object_primary(env_with_assembler, "assembler")
        other_layer = 1 if layer == 0 else 0
        height = env_with_assembler.map_height
        width = env_with_assembler.map_width

        occupied = set()
        objects = env_with_assembler.grid_objects()
        for obj in objects.values():
            for c_loc, r_loc, layer_loc in obj.get("locations", []):
                occupied.add((r_loc, c_loc, layer_loc))

        adjacent_target = None
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= height:
                continue
            if nc < 0 or nc >= width:
                continue
            if (nr, nc, layer) in occupied:
                continue
            adjacent_target = (nr, nc)
            break
        assert adjacent_target is not None, "Expected a free adjacent cell near assembler for test setup"

        ok = env_with_assembler.add_object_location(assembler_id, adjacent_target[0], adjacent_target[1], other_layer)
        assert not ok, "Adding locations on a different layer should be rejected"

    def test_assembler_remove_object_cell_rejects_last_cell(self, env_with_assembler):
        env_with_assembler.reset()

        assembler_id, (r, c, layer) = _find_object_primary(env_with_assembler, "assembler")

        removed = env_with_assembler.remove_object_location(assembler_id, r, c, layer)
        assert not removed, "Removing the last cell should be rejected"

    def test_assembler_supports_large_shapes(self, env_with_assembler):
        env_with_assembler.reset()

        assembler_id, (r, c, layer) = _find_object_primary(env_with_assembler, "assembler")

        directions = ((0, 1), (1, 0), (0, -1), (-1, 0))
        height = env_with_assembler.map_height
        width = env_with_assembler.map_width

        target_total = 12  # 11 extra locations beyond primary location
        visited = {(r, c)}
        queue = [(r, c)]
        attempted = {(r, c)}

        while queue and len(visited) < target_total:
            base_r, base_c = queue.pop(0)
            for dr, dc in directions:
                nr = base_r + dr
                nc = base_c + dc
                if nr < 0 or nr >= height:
                    continue
                if nc < 0 or nc >= width:
                    continue
                if (nr, nc) in attempted:
                    continue
                attempted.add((nr, nc))
                ok = env_with_assembler.add_object_location(assembler_id, nr, nc, layer)
                if ok:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
                if len(visited) >= target_total:
                    break

        assembler = env_with_assembler.grid_objects()[assembler_id]
        locations = assembler.get("locations", [])
        assert "max_locations" not in assembler, "max_locations should no longer be exposed"
        assert len(locations) == len(visited), "Reported locations should match locations tracked in the environment"
        assert len(locations) == len({(rr, cc) for cc, rr, _ in locations}), "Cells should remain unique"
        assert len(locations) >= 8, "Assembler should be able to occupy well beyond 5 locations"


class TestChestProperties:
    """Test chest-specific properties in grid_objects."""

    def test_chest_basic_properties(self, env_with_chest):
        """Test that chest properties are exposed."""
        env_with_chest.reset()

        objects = env_with_chest.grid_objects()

        # Find a chest
        chest = next((obj for obj in objects.values() if obj.get("type_name") == "chest"), None)

        if chest:
            # Check chest-specific properties
            assert "resource_type" in chest
            assert "position_deltas" in chest
            assert "max_inventory" in chest

            # Check values match config
            assert chest["resource_type"] == 0, "Should be gold (resource 0)"
            # Positions are stored as integers internally (bit indices)
            # NW=0, N=1, NE=2, SW=5, S=6, SE=7
            # Positive delta = deposit, negative delta = withdraw
            position_deltas = chest["position_deltas"]
            assert position_deltas[0] == 1, "NW should have delta +1 (deposit)"
            assert position_deltas[1] == 1, "N should have delta +1 (deposit)"
            assert position_deltas[2] == 1, "NE should have delta +1 (deposit)"
            assert position_deltas[5] == -1, "SW should have delta -1 (withdraw)"
            assert position_deltas[6] == -1, "S should have delta -1 (withdraw)"
            assert position_deltas[7] == -1, "SE should have delta -1 (withdraw)"

            # Check that chest has inventory dict
            assert "inventory" in chest
            assert isinstance(chest["inventory"], dict)


class TestAgentProperties:
    """Test that all agent properties are properly exposed."""

    def test_agent_properties(self, env_with_walls):
        """Test that all agent properties are exposed."""
        env_with_walls.reset()

        objects = env_with_walls.grid_objects()

        # Find an agent
        agent = next((obj for obj in objects.values() if obj.get("type_name") == "agent"), None)

        assert agent is not None, "Should have at least one agent"

        # Check all agent properties
        required_properties = [
            "agent_id",
            "orientation",
            "group_id",
            "is_frozen",
            "freeze_remaining",
            "freeze_duration",
            "inventory",
        ]

        for prop in required_properties:
            assert prop in agent, f"Agent should have {prop} property"

        # Check types
        assert isinstance(agent["agent_id"], (int, np.integer))
        assert isinstance(agent["orientation"], (int, np.integer))
        assert isinstance(agent["group_id"], (int, np.integer))
        assert isinstance(agent["is_frozen"], bool)
        assert isinstance(agent["freeze_remaining"], (int, np.integer))
        assert isinstance(agent["freeze_duration"], (int, np.integer))
        assert isinstance(agent["inventory"], dict)
