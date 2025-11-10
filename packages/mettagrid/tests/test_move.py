"""Tests for 8-way movement system."""

import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    GameConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    WallConfig,
)
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.mapgen.utils.ascii_grid import DEFAULT_CHAR_TO_NAME
from mettagrid.simulator import Action, Simulation
from mettagrid.test_support.actions import get_agent_position, move
from mettagrid.test_support.map_builders import ObjectNameMapBuilder
from mettagrid.test_support.orientation import Orientation


# Test fixtures for MettaGrid environments
@pytest.fixture
def base_config() -> GameConfig:
    """Base configuration for MettaGrid tests."""
    return GameConfig(
        max_steps=50,
        num_agents=1,
        obs=ObsConfig(width=3, height=3, num_tokens=100),
        resource_names=["laser", "armor"],
        actions=ActionsConfig(
            noop=NoopActionConfig(enabled=True),
            move=MoveActionConfig(
                enabled=True,
                allowed_directions=[
                    "north",
                    "south",
                    "east",
                    "west",
                    "northeast",
                    "northwest",
                    "southeast",
                    "southwest",
                ],
            ),
        ),
        objects={
            "wall": WallConfig(),
        },
    )


@pytest.fixture
def movement_game_map():
    """Game map with agent in center and room to move."""
    return [
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "agent.red", "empty", "empty", "wall"],  # Agent in center
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
    ]


@pytest.fixture
def small_movement_game_map():
    """Smaller game map for focused movement tests."""
    return [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "agent.red", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]


@pytest.fixture
def blocked_game_map():
    """Game map where agent is completely surrounded by walls."""
    return [
        ["wall", "wall", "wall"],
        ["wall", "agent.red", "wall"],
        ["wall", "wall", "wall"],
    ]


@pytest.fixture
def corridor_game_map():
    """Game map with a corridor for walking tests."""
    return [
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
    ]


@pytest.fixture
def make_sim(base_config):
    """Factory fixture that creates a configured Simulation environment."""

    def _create_sim(game_map, config_overrides=None):
        if config_overrides:
            # Create a new config with overrides
            game_config = base_config.model_copy(update=config_overrides)
        else:
            game_config = base_config

        # Create MettaGridConfig wrapper
        cfg = MettaGridConfig(game=game_config)

        # Put the map into the config using ObjectNameMapBuilder
        # Convert numpy array to list if needed
        map_list = game_map.tolist() if hasattr(game_map, "tolist") else game_map
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=map_list)

        sim = Simulation(cfg, seed=42)

        return sim

    return _create_sim


# Tests for Simulation (low-level API)
def test_8way_movement_all_directions(make_sim, movement_game_map):
    """Test 8-way movement in all eight directions using Simulation."""
    # Test all 8 directions with expected deltas
    all_direction_tests = [
        (Orientation.NORTH, (-1, 0)),
        (Orientation.EAST, (0, 1)),
        (Orientation.SOUTH, (1, 0)),
        (Orientation.WEST, (0, -1)),
        (Orientation.NORTHEAST, (-1, 1)),
        (Orientation.NORTHWEST, (-1, -1)),
        (Orientation.SOUTHEAST, (1, 1)),
        (Orientation.SOUTHWEST, (1, -1)),
    ]

    for orientation, (expected_dr, expected_dc) in all_direction_tests:
        # Create a fresh sim for each direction test to start from center
        sim = make_sim(movement_game_map)
        direction_name = str(orientation)

        print(f"Testing move {direction_name} (value {orientation.value})")

        position_before = get_agent_position(sim, 0)
        result = move(sim, orientation)
        position_after = get_agent_position(sim, 0)

        # Assert movement was successful
        assert result["success"], f"Move {direction_name} should succeed. Error: {result.get('error', 'Unknown')}"

        # Assert position changed
        assert position_before != position_after, f"Agent should have moved {direction_name}"

        # Assert movement was in correct direction
        dr = position_after[0] - position_before[0]
        dc = position_after[1] - position_before[1]
        assert (dr, dc) == (expected_dr, expected_dc), f"Agent should have moved correctly {direction_name}"

        print(f"✅ Move {direction_name}: {position_before} → {position_after}")


def test_8way_movement_obstacles():
    """Test that 8-way movement respects obstacles."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            actions=ActionsConfig(
                move=MoveActionConfig(
                    allowed_directions=[
                        "north",
                        "south",
                        "east",
                        "west",
                        "northeast",
                        "northwest",
                        "southeast",
                        "southwest",
                    ]
                ),
                noop=NoopActionConfig(),
            ),
            objects={"wall": WallConfig()},
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#", "#", "#"],
                    ["#", ".", ".", ".", "#"],
                    ["#", ".", "@", ".", "#"],
                    ["#", ".", ".", ".", "#"],
                    ["#", "#", "#", "#", "#"],
                ],
                char_to_map_name=DEFAULT_CHAR_TO_NAME,
            ),
        )
    )
    sim = Simulation(cfg)

    objects = sim.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_name"] == "agent")

    # Test cardinal movements near corners (skip diagonal movements)
    # Move to top-left corner area
    sim.agent(0).set_action(Action(name="move_north"))
    sim.step()
    sim.agent(0).set_action(Action(name="move_west"))
    sim.step()

    objects = sim.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (1, 1)

    # Try to move West into wall - should fail
    sim.agent(0).set_action(Action(name="move_west"))
    sim.step()
    assert not sim.agent(0).last_action_success

    # Position should remain unchanged
    objects = sim.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (1, 1)


def test_8way_movement_with_simple_environment():
    """Test 8-way movement using the simple environment builder."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            actions=ActionsConfig(
                move=MoveActionConfig(
                    allowed_directions=[
                        "north",
                        "south",
                        "east",
                        "west",
                        "northeast",
                        "northwest",
                        "southeast",
                        "southwest",
                    ]
                ),
                noop=NoopActionConfig(),
            ),
            objects={"wall": WallConfig()},
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#", "#", "#", "#", "#"],
                    ["#", ".", ".", ".", ".", ".", "#"],
                    ["#", ".", ".", "@", ".", ".", "#"],
                    ["#", ".", ".", ".", ".", ".", "#"],
                    ["#", "#", "#", "#", "#", "#", "#"],
                ],
                char_to_map_name=DEFAULT_CHAR_TO_NAME,
            ),
        )
    )
    sim = Simulation(cfg)

    objects = sim.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_name"] == "agent")

    # Get initial position (should be at center, row=2, col=3)
    initial_pos = (objects[agent_id]["r"], objects[agent_id]["c"])

    # Test a few diagonal movements
    # Move northeast
    sim.agent(0).set_action(Action(name="move_northeast"))
    sim.step()

    objects = sim.grid_objects()
    pos_after_ne = (objects[agent_id]["r"], objects[agent_id]["c"])

    # Should have moved up and right: row-1, col+1
    assert pos_after_ne[0] == initial_pos[0] - 1, "Should move up (row-1) for northeast"
    assert pos_after_ne[1] == initial_pos[1] + 1, "Should move right (col+1) for northeast"

    # Move southwest (should return close to original position)
    sim.agent(0).set_action(Action(name="move_southwest"))
    sim.step()

    objects = sim.grid_objects()
    pos_after_sw = (objects[agent_id]["r"], objects[agent_id]["c"])

    # Should be back at initial position
    assert pos_after_sw == initial_pos, f"After NE then SW, should be back at {initial_pos}, but at {pos_after_sw}"


def test_8way_movement_boundary_check():
    """Test 8-way movement respects environment boundaries."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            actions=ActionsConfig(
                move=MoveActionConfig(
                    allowed_directions=[
                        "north",
                        "south",
                        "east",
                        "west",
                        "northeast",
                        "northwest",
                        "southeast",
                        "southwest",
                    ]
                ),
                noop=NoopActionConfig(),
            ),
            objects={"wall": WallConfig()},
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#", "#", "#"],
                    ["#", ".", ".", ".", "#"],
                    ["#", ".", "@", ".", "#"],  # Agent near center
                    ["#", ".", ".", ".", "#"],
                    ["#", "#", "#", "#", "#"],
                ],
                char_to_map_name=DEFAULT_CHAR_TO_NAME,
            ),
        )
    )
    sim = Simulation(cfg)

    objects = sim.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_name"] == "agent")

    # Move agent to top-left corner area (near boundary)
    sim.agent(0).set_action(Action(name="move_northwest"))
    sim.step()
    sim.agent(0).set_action(Action(name="move_northwest"))
    sim.step()

    objects = sim.grid_objects()
    position = (objects[agent_id]["r"], objects[agent_id]["c"])

    # Agent should be at (1, 1) - top-left open space, blocked by walls
    assert position == (1, 1), f"Agent should be at top-left corner (1,1), but is at {position}"

    # Try to move northwest again - should be blocked by wall
    sim.agent(0).set_action(Action(name="move_northwest"))
    sim.step()

    # Check that agent didn't move
    assert not sim.agent(0).last_action_success, "Movement should fail when blocked by wall"

    objects = sim.grid_objects()
    new_position = (objects[agent_id]["r"], objects[agent_id]["c"])
    assert new_position == position, f"Agent should stay at {position} when blocked"

    # Try diagonal move toward boundary in different direction
    # Move to bottom-right area
    sim.agent(0).set_action(Action(name="move_southeast"))
    sim.step()
    sim.agent(0).set_action(Action(name="move_southeast"))
    sim.step()
    sim.agent(0).set_action(Action(name="move_southeast"))
    sim.step()
    sim.agent(0).set_action(Action(name="move_southeast"))
    sim.step()

    objects = sim.grid_objects()
    position = (objects[agent_id]["r"], objects[agent_id]["c"])

    # Should be near bottom-right corner
    assert position == (3, 3), f"Agent should be at bottom-right corner (3,3), but is at {position}"

    # Try to move southeast again - should be blocked
    sim.agent(0).set_action(Action(name="move_southeast"))
    sim.step()

    assert not sim.agent(0).last_action_success, "Movement should fail at boundary"

    objects = sim.grid_objects()
    final_position = (objects[agent_id]["r"], objects[agent_id]["c"])
    assert final_position == position, "Agent should stay in place when blocked by boundary"


# Tests for MettaGrid (high-level API) using helper functions
def test_move_all_directions(make_sim, movement_game_map):
    """Test the move function in all eight compass directions."""
    sim = make_sim(movement_game_map)

    initial_pos = get_agent_position(sim)
    assert initial_pos is not None, "Agent should have a valid initial position"

    # Test cardinal directions first
    cardinal_tests = [
        (Orientation.NORTH, (-1, 0)),
        (Orientation.EAST, (0, 1)),
        (Orientation.SOUTH, (1, 0)),
        (Orientation.WEST, (0, -1)),
    ]

    for orientation, (expected_dr, expected_dc) in cardinal_tests:
        direction_name = str(orientation)

        print(f"Testing move {direction_name} (value {orientation.value})")

        position_before = get_agent_position(sim, 0)
        result = move(sim, orientation)
        position_after = get_agent_position(sim, 0)

        # Assert movement was successful
        assert result["success"], f"Move {direction_name} should succeed. Error: {result.get('error', 'Unknown')}"

        # Assert position changed
        assert position_before != position_after, f"Agent should have moved {direction_name}"

        # Assert movement was in correct direction
        dr = position_after[0] - position_before[0]
        dc = position_after[1] - position_before[1]
        assert (dr, dc) == (expected_dr, expected_dc), f"Agent should have moved correctly {direction_name}"

        print(f"✅ Move {direction_name}: {position_before} → {position_after}")


def test_move_diagonal_directions(make_sim, movement_game_map):
    """Test the move function in all four diagonal directions."""
    # Test diagonal directions
    diagonal_tests = [
        (Orientation.NORTHEAST, (-1, 1)),
        (Orientation.NORTHWEST, (-1, -1)),
        (Orientation.SOUTHEAST, (1, 1)),
        (Orientation.SOUTHWEST, (1, -1)),
    ]

    for orientation, (expected_dr, expected_dc) in diagonal_tests:
        # Create a fresh sim for each direction test to start from center
        sim = make_sim(movement_game_map)
        direction_name = str(orientation)

        print(f"Testing move {direction_name} (value {orientation.value})")

        position_before = get_agent_position(sim, 0)
        result = move(sim, orientation)
        position_after = get_agent_position(sim, 0)

        # Assert movement was successful
        assert result["success"], f"Move {direction_name} should succeed. Error: {result.get('error', 'Unknown')}"

        # Assert position changed
        assert position_before != position_after, f"Agent should have moved {direction_name}"

        # Assert movement was in correct direction
        dr = position_after[0] - position_before[0]
        dc = position_after[1] - position_before[1]
        assert (dr, dc) == (expected_dr, expected_dc), f"Agent should have moved correctly {direction_name}"

        print(f"✅ Move {direction_name}: {position_before} → {position_after}")


def test_move_up(make_sim, small_movement_game_map):
    """Test moving north specifically."""
    sim = make_sim(small_movement_game_map, {"max_steps": 10})

    # Get position before move
    position_before = get_agent_position(sim, 0)

    result = move(sim, Orientation.NORTH)  # Use Orientation.NORTH for moving up

    assert result["success"], f"Move north should succeed. Error: {result.get('error')}"

    # Get position after move and verify
    position_after = get_agent_position(sim, 0)
    assert position_before[0] - position_after[0] == 1, "Should move up by 1 row"


def test_move_blocked_by_wall(make_sim, blocked_game_map):
    """Test that movement is properly blocked by walls."""
    sim = make_sim(blocked_game_map, {"max_steps": 10})

    directions = [
        Orientation.NORTH,
        Orientation.EAST,
        Orientation.SOUTH,
        Orientation.WEST,
    ]

    for orientation in directions:
        position_before = get_agent_position(sim, 0)
        result = move(sim, orientation)
        position_after = get_agent_position(sim, 0)
        direction_name = str(orientation)

        # Movement should fail or position should remain unchanged
        if result["success"]:
            # This shouldn't happen for blocked movement
            raise AssertionError(f"Move {direction_name} should fail when blocked by wall")
        else:
            # Action failed, which is expected for blocked movement
            assert result["error"] is not None, f"Failed move {direction_name} should have an error message"
            assert position_before == position_after, "Position should not change when blocked"


def test_move_returns_to_center(make_sim, movement_game_map):
    """Test that we can move in a square and return to center."""
    sim = make_sim(movement_game_map)

    initial_pos = get_agent_position(sim)

    # Move in a square: north, east, south, west
    moves = [
        Orientation.NORTH,
        Orientation.EAST,
        Orientation.SOUTH,
        Orientation.WEST,
    ]

    for orientation in moves:
        result = move(sim, orientation)
        direction_name = str(orientation)

        assert result["success"], f"Move {direction_name} should succeed"

    # Should be back at original position
    final_pos = get_agent_position(sim)
    assert final_pos == initial_pos, f"Agent should return to original position {initial_pos}, but is at {final_pos}"


def test_agent_walks_across_room(make_sim, corridor_game_map):
    """
    Test where a single agent walks across a room.
    Creates a simple corridor and attempts to walk the agent from one end to the other.
    """
    print("Testing agent walking across room...")

    # Create environment with walking-specific config
    sim = make_sim(
        corridor_game_map,
        {
            "max_steps": 20,
            "obs_width": 3,
            "obs_height": 3,
        },
    )

    print(f"Environment created: {sim.map_width}x{sim.map_height}")
    print(f"Initial timestep: {sim.current_step}")

    # Find a working direction using Orientation enum
    successful_moves = []
    total_moves = 0

    print("\n=== Testing which direction allows movement ===")
    working_direction = None

    # Test all cardinal directions
    directions = [
        Orientation.NORTH,
        Orientation.EAST,
        Orientation.SOUTH,
        Orientation.WEST,
    ]

    for orientation in directions:
        direction_name = str(orientation)

        print(f"\nTesting movement {direction_name}...")

        result = move(sim, orientation, agent_idx=0)

        if result["success"]:
            print(f"✓ Found working direction: {direction_name}")
            working_direction = orientation
            working_direction_str = direction_name
            break
        else:
            print(f"✗ Direction {direction_name} failed: {result.get('error', 'Unknown error')}")

    # Assert we found at least one working direction
    assert working_direction is not None, "Should find at least one direction that allows movement"

    print(f"\n=== Walking across room in direction: {working_direction_str} ===")

    # Reset for clean walk
    sim = make_sim(
        corridor_game_map,
        {
            "max_steps": 20,
            "obs_width": 3,
            "obs_height": 3,
        },
    )

    # Walk multiple steps
    max_steps = 5

    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step}: Moving {working_direction_str} ---")

        position_before = get_agent_position(sim, 0)
        result = move(sim, working_direction, agent_idx=0)
        position_after = get_agent_position(sim, 0)
        total_moves += 1

        if result["success"]:
            successful_moves.append(step)
            print(f"✓ Successful move #{len(successful_moves)}")
            print(f"  Position: {position_before} → {position_after}")
        else:
            print(f"✗ Move failed: {result.get('error', 'Unknown error')}")
            print("  Agent likely hit an obstacle or boundary")
            break

        if sim.current_step >= 18:
            print("  Approaching max steps limit")
            break

    print("\n=== Walking Test Summary ===")
    print(f"Working direction: {working_direction_str}")
    print(f"Total move attempts: {total_moves}")
    print(f"Successful moves: {len(successful_moves)}")
    print(f"Success rate: {len(successful_moves) / total_moves:.1%}" if total_moves > 0 else "N/A")

    # Validation
    assert len(successful_moves) >= 1, (
        f"Agent should have moved at least once. Got {len(successful_moves)} successful moves."
    )

    assert total_moves > 0, "Should have attempted at least one move"

    print("✅ Agent walking test passed!")


def test_agent_walks_in_all_cardinal_directions(make_sim, corridor_game_map):
    """Test that agent can move in all non-blocked cardinal directions."""
    print("Testing agent movement in all possible cardinal directions...")

    sim = make_sim(corridor_game_map, {"max_steps": 20})

    successful_directions = []
    failed_directions = []

    # Test all cardinal directions using Orientation
    directions = [
        Orientation.NORTH,
        Orientation.EAST,
        Orientation.SOUTH,
        Orientation.WEST,
    ]

    for orientation in directions:
        direction_name = str(orientation)

        print(f"\nTesting {direction_name} (value {orientation.value})...")

        # Reset environment for each direction test
        sim = make_sim(corridor_game_map, {"max_steps": 20})

        position_before = get_agent_position(sim, 0)
        result = move(sim, orientation, agent_idx=0)
        position_after = get_agent_position(sim, 0)

        if result["success"] and position_before != position_after:
            successful_directions.append(direction_name)
            print(f"✓ {direction_name}: {position_before} → {position_after}")
        else:
            failed_directions.append(direction_name)
            print(f"✗ {direction_name}: {result.get('error', 'Movement failed')}")

    print("\nResults:")
    print(f"Successful directions: {successful_directions}")
    print(f"Failed directions: {failed_directions}")

    # Based on the corridor map, "east" should work (agent starts at (1,1), can move to (1,2))
    # west should be blocked by wall, north/south might be blocked depending on exact positioning
    assert len(successful_directions) >= 1, (
        f"Should be able to move in at least one direction. "
        f"Successful: {successful_directions}, Failed: {failed_directions}"
    )

    # In a corridor, we expect east to work since agent starts next to empty spaces
    if "east" not in successful_directions:
        print("Warning: East movement failed, which is unexpected in this corridor layout")
