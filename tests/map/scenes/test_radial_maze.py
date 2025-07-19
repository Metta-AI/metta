from metta.map.scenes.radial_maze import RadialMaze
from tests.map.scenes.utils import render_scene


def test_basic():
    """Test basic functionality of RadialMaze scene."""
    scene = render_scene(RadialMaze, {"arms": 4, "arm_width": 2, "arm_length": 5}, (15, 15))

    # Center should be at the middle of the grid
    cx, cy = 7, 7  # center of 15x15 grid

    # Check that there are both walls and empty spaces
    wall_count = (scene.grid == "wall").sum()
    empty_count = (scene.grid == "empty").sum()

    assert wall_count > 0
    assert empty_count > 0

    # Center should be empty (part of the maze)
    assert scene.grid[cy, cx] == "empty"


def test_different_arm_counts():
    """Test radial maze with different numbers of arms."""
    empty_counts = []

    for arms in [4, 6, 8]:
        scene = render_scene(RadialMaze, {"arms": arms, "arm_width": 1, "arm_length": 4}, (13, 13))

        # Should have both walls and empty spaces
        wall_count = (scene.grid == "wall").sum()
        empty_count = (scene.grid == "empty").sum()
        empty_counts.append(empty_count)

        assert wall_count > 0
        assert empty_count > 0

        # Center should be accessible
        cx, cy = 6, 6  # center of 13x13 grid
        assert scene.grid[cy, cx] == "empty"

    # More arms should create more empty space
    assert empty_counts[1] > empty_counts[0]  # 6 arms > 4 arms
    assert empty_counts[2] > empty_counts[1]  # 8 arms > 6 arms


def test_arm_width_variations():
    """Test different arm widths."""
    empty_counts = []

    # Note: width=1 and width=2 (or 3 and 4) are the same.
    for arm_width in [2, 4, 6]:
        scene = render_scene(RadialMaze, {"arms": 5, "arm_width": arm_width, "arm_length": 20}, (45, 45))

        empty_count = (scene.grid == "empty").sum()
        empty_counts.append(empty_count)
        assert empty_count > 0

        # Center should be accessible
        cx, cy = 22, 22
        assert scene.grid[cy, cx] == "empty"

    # Wider arms should create more empty space
    assert empty_counts[1] > empty_counts[0]  # arm_width=4 > arm_width=2
    assert empty_counts[2] > empty_counts[1]  # arm_width=6 > arm_width=4


def test_large_maze():
    """Test radial maze with larger dimensions."""
    scene = render_scene(RadialMaze, {"arms": 6, "arm_width": 3, "arm_length": 8}, (25, 25))

    # Should have reasonable proportion of walls and empty spaces
    total_cells = 25 * 25
    wall_count = (scene.grid == "wall").sum()
    empty_count = (scene.grid == "empty").sum()

    assert wall_count + empty_count == total_cells
    assert empty_count > 0
    assert wall_count > 0

    # Center should be accessible
    cx, cy = 12, 12  # center of 25x25 grid
    assert scene.grid[cy, cx] == "empty"


def test_auto_arm_length():
    """Test radial maze with automatic arm length calculation."""
    scene = render_scene(
        RadialMaze,
        {
            "arms": 4,
            "arm_width": 1,
            "arm_length": None,  # Should auto-calculate
        },
        (11, 11),
    )

    # Should still create a valid maze
    assert (scene.grid == "wall").sum() > 0
    assert (scene.grid == "empty").sum() > 0

    # Center should be accessible
    cx, cy = 5, 5  # center of 11x11 grid
    assert scene.grid[cy, cx] == "empty"


def test_minimum_arms():
    """Test with minimum number of arms."""
    scene = render_scene(
        RadialMaze,
        {
            "arms": 4,  # minimum allowed
            "arm_width": 1,
            "arm_length": 3,
        },
        (9, 9),
    )

    # Should create a valid maze
    assert (scene.grid == "wall").sum() > 0
    assert (scene.grid == "empty").sum() > 0


def test_maximum_arms():
    """Test with maximum number of arms."""
    scene = render_scene(
        RadialMaze,
        {
            "arms": 12,  # maximum allowed
            "arm_width": 1,
            "arm_length": 4,
        },
        (17, 17),
    )

    # Should create a valid maze
    assert (scene.grid == "wall").sum() > 0
    assert (scene.grid == "empty").sum() > 0

    # With many arms, should have substantial empty space
    empty_count = (scene.grid == "empty").sum()
    assert empty_count > 20  # Should have decent amount of empty space


def test_small_grid():
    """Test radial maze on a small grid."""
    scene = render_scene(RadialMaze, {"arms": 4, "arm_width": 1, "arm_length": 2}, (7, 7))

    # Should still create some structure
    assert (scene.grid == "wall").sum() > 0
    assert (scene.grid == "empty").sum() > 0

    # Center should be accessible
    cx, cy = 3, 3  # center of 7x7 grid
    assert scene.grid[cy, cx] == "empty"
