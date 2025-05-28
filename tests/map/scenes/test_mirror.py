from metta.map.scenes.maze import MazeKruskal
from metta.map.scenes.mirror import Mirror
from tests.map.scenes.utils import check_grid, scene_to_node


def test_horizontal():
    scene = Mirror(scene=MazeKruskal(seed=123), symmetry="horizontal")
    node = scene_to_node(scene, (9, 9))

    check_grid(
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
    scene = Mirror(scene=MazeKruskal(seed=123), symmetry="vertical")
    node = scene_to_node(scene, (9, 9))

    check_grid(
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
    scene = Mirror(scene=MazeKruskal(seed=123), symmetry="x4")
    node = scene_to_node(scene, (9, 9))

    check_grid(
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
