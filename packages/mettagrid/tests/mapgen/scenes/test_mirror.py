from mettagrid.mapgen.scenes.maze import Maze
from mettagrid.mapgen.scenes.mirror import Mirror
from mettagrid.test_support.mapgen import assert_grid, render_scene


def test_horizontal():
    scene = render_scene(
        Mirror.factory(Mirror.Params(scene=Maze.factory(seed=123), symmetry="horizontal")),
        (9, 9),
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
        Mirror.factory(Mirror.Params(scene=Maze.factory(seed=123), symmetry="vertical")),
        (9, 9),
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
        Mirror.factory(Mirror.Params(scene=Maze.factory(seed=123), symmetry="x4")),
        (9, 9),
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
