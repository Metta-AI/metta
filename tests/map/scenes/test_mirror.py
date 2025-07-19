from metta.map.scenes.maze import Maze
from metta.map.scenes.mirror import Mirror
from tests.map.scenes.utils import assert_grid, render_scene


def test_horizontal():
    scene = render_scene(
        Mirror,
        {"scene": Maze.factory({}, seed=123), "symmetry": "horizontal"},
        shape=(9, 9),
    )

    assert_grid(
        scene,
        """
           .#.#.#.#.
           .#.#.#.#.
           .#.....#.
           .###.###.
           .#.....#.
           .#.#.#.#.
           ...#.#...
           ##.###.##
           .........
        """,
    )


def test_vertical():
    scene = render_scene(
        Mirror,
        {"scene": Maze.factory({}, seed=123), "symmetry": "vertical"},
        shape=(9, 9),
    )

    assert_grid(
        scene,
        """
           ...#.....
           ##.#.#.#.
           ...#.#.#.
           .#####.##
           .........
           .#####.##
           ...#.#.#.
           ##.#.#.#.
           ...#.....
        """,
    )


def test_x4():
    scene = render_scene(
        Mirror,
        {"scene": Maze.factory({}, seed=123), "symmetry": "x4"},
        shape=(9, 9),
    )

    scene.print_scene_tree()

    assert_grid(
        scene,
        """
           .#.#.#.#.
           .#.#.#.#.
           .........
           .#######.
           .........
           .#######.
           .........
           .#.#.#.#.
           .#.#.#.#.
        """,
    )
