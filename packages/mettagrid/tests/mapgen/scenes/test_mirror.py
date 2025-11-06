import mettagrid.mapgen.scenes.maze
import mettagrid.mapgen.scenes.mirror
import mettagrid.test_support.mapgen
import packages.mettagrid.tests.mapgen.scenes.test_utils


def test_horizontal():
    scene = mettagrid.test_support.mapgen.render_scene(
        mettagrid.mapgen.scenes.mirror.Mirror.Config(
            scene=mettagrid.mapgen.scenes.maze.Maze.Config(seed=123), symmetry="horizontal"
        ),
        (9, 9),
    )

    packages.mettagrid.tests.mapgen.scenes.test_utils.assert_grid(
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
    scene = mettagrid.test_support.mapgen.render_scene(
        mettagrid.mapgen.scenes.mirror.Mirror.Config(
            scene=mettagrid.mapgen.scenes.maze.Maze.Config(seed=123), symmetry="vertical"
        ),
        (9, 9),
    )

    packages.mettagrid.tests.mapgen.scenes.test_utils.assert_grid(
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
    scene = mettagrid.test_support.mapgen.render_scene(
        mettagrid.mapgen.scenes.mirror.Mirror.Config(
            scene=mettagrid.mapgen.scenes.maze.Maze.Config(seed=123), symmetry="x4"
        ),
        (9, 9),
    )

    packages.mettagrid.tests.mapgen.scenes.test_utils.assert_grid(
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
