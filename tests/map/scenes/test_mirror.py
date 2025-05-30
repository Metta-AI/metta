from metta.map.scenes.maze import MazeKruskal
from metta.map.scenes.mirror import Mirror
from tests.map.scenes.utils import assert_grid, render_node


def test_horizontal():
    node = render_node(
        Mirror,
        {"scene": lambda grid: MazeKruskal(grid=grid, params={}, seed=123), "symmetry": "horizontal"},
        shape=(9, 9),
    )

    assert_grid(
        node,
        """
           | # # # # |
           | # # # # |
           | #     # |
           | ### ### |
           | #     # |
           | # # # # |
           |   # #   |
           |## ### ##|
           |         |
        """,
    )


def test_vertical():
    node = render_node(
        Mirror,
        {"scene": lambda grid: MazeKruskal(grid=grid, params={}, seed=123), "symmetry": "vertical"},
        shape=(9, 9),
    )

    assert_grid(
        node,
        """
           |   #     |
           |## # # # |
           |   # # # |
           | ##### ##|
           |         |
           | ##### ##|
           |   # # # |
           |## # # # |
           |   #     |
        """,
    )


def test_x4():
    node = render_node(
        Mirror,
        {"scene": lambda grid: MazeKruskal(grid=grid, params={}, seed=123), "symmetry": "x4"},
        shape=(9, 9),
    )

    assert_grid(
        node,
        """
           | # # # # |
           | # # # # |
           |         |
           | ####### |
           |         |
           | ####### |
           |         |
           | # # # # |
           | # # # # |
        """,
    )
