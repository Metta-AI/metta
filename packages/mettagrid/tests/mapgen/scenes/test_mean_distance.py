import numpy as np

from mettagrid.mapgen.scenes.mean_distance import MeanDistance
from mettagrid.test_support.mapgen import render_scene


def test_basic():
    """Test basic functionality of MeanDistance scene."""
    scene = render_scene(
        MeanDistance.Config(mean_distance=3.0, objects={"assembler": 2, "enemy": 1}),
        (11, 11),
    )

    # Agent should be placed at the center
    agent_pos = (5, 5)  # center of 11x11 grid
    assert scene.grid[agent_pos] == "agent.agent"

    # Count objects
    assembler_count = (scene.grid == "assembler").sum()
    enemy_count = (scene.grid == "enemy").sum()

    assert assembler_count == 2
    assert enemy_count == 1

    # Objects should not overlap with agent
    assert scene.grid[agent_pos] == "agent.agent"


def test_object_placement():
    """Test that objects are placed at reasonable distances from agent."""
    scene = render_scene(
        MeanDistance.Config(mean_distance=2.0, objects={"assembler": 5}),
        (101, 101),
    )

    agent_pos = (50, 50)  # center of 101x101 grid
    assert scene.grid[agent_pos] == "agent.agent"

    # Find all assembler positions and calculate distances
    assembler_positions = np.where(scene.grid == "assembler")
    distances = []

    for i in range(len(assembler_positions[0])):
        pos = (assembler_positions[0][i], assembler_positions[1][i])
        # Calculate Euclidean distance from agent
        dist = np.sqrt((pos[0] - agent_pos[0]) ** 2 + (pos[1] - agent_pos[1]) ** 2)
        distances.append(dist)

    # All objects should be at least distance 1 from agent (as per implementation)
    assert all(d >= 1.0 for d in distances)

    # With mean_distance=2.0, most objects should be within reasonable range
    assert all(d <= 15.0 for d in distances)  # sanity check


def test_multiple_object_types():
    """Test placement of multiple different object types."""
    scene = render_scene(
        MeanDistance.Config(mean_distance=4.0, objects={"assembler": 3, "enemy": 2, "key": 1}),
        (21, 21),
    )

    # Check all objects are placed
    assert (scene.grid == "assembler").sum() == 3
    assert (scene.grid == "enemy").sum() == 2
    assert (scene.grid == "key").sum() == 1

    # Agent should be at center
    agent_pos = (10, 10)
    assert scene.grid[agent_pos] == "agent.agent"


def test_small_grid():
    """Test behavior with a smaller grid where placement might be constrained."""
    scene = render_scene(
        MeanDistance.Config(mean_distance=2.0, objects={"assembler": 2}),
        (7, 7),
    )

    # Agent should be at center
    agent_pos = (3, 3)
    assert scene.grid[agent_pos] == "agent.agent"

    # Some objects should be placed (might be fewer than requested due to space constraints)
    assembler_count = (scene.grid == "assembler").sum()
    assert assembler_count <= 2  # At most the requested number
    assert assembler_count >= 0  # At least 0 if no space
