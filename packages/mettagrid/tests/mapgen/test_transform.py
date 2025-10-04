import numpy as np

from mettagrid.map_builder.utils import create_grid
from mettagrid.mapgen.area import AreaQuery, AreaWhere
from mettagrid.mapgen.scene import ChildrenAction, GridTransform, Scene, SceneConfig
from mettagrid.test_support.mapgen import assert_raw_grid, render_scene


class SerifConfig(SceneConfig):
    pass


class SerifScene(Scene[SerifConfig]):
    def render(self):
        self.grid[1, 0] = "wall"
        self.make_area(1, 1, self.width - 2, self.height - 2, tags=["inner"])


class TestGridTransformClass:
    def test_identity(self):
        assert GridTransform.IDENTITY.compose(GridTransform.IDENTITY) == GridTransform.IDENTITY
        assert GridTransform.IDENTITY.compose(GridTransform.ROT_90) == GridTransform.ROT_90

    def test_rot_composition(self):
        assert GridTransform.ROT_90.compose(GridTransform.ROT_90) == GridTransform.ROT_180
        assert GridTransform.ROT_90.compose(GridTransform.ROT_180) == GridTransform.ROT_270
        assert GridTransform.ROT_90.compose(GridTransform.ROT_270) == GridTransform.IDENTITY

    def test_inverse(self):
        assert GridTransform.IDENTITY.inverse() == GridTransform.IDENTITY
        assert GridTransform.ROT_90.inverse() == GridTransform.ROT_270
        assert GridTransform.ROT_180.inverse() == GridTransform.ROT_180
        assert GridTransform.ROT_270.inverse() == GridTransform.ROT_90
        assert GridTransform.FLIP_H.inverse() == GridTransform.FLIP_H
        assert GridTransform.FLIP_V.inverse() == GridTransform.FLIP_V
        assert GridTransform.TRANSPOSE_ALT.inverse() == GridTransform.TRANSPOSE_ALT
        assert GridTransform.TRANSPOSE.inverse() == GridTransform.TRANSPOSE

    def test_apply(self):
        grid = np.array([[0, 1, 2], [3, 4, 5]])
        np.testing.assert_array_equal(GridTransform.ROT_90.apply(grid), np.array([[3, 0], [4, 1], [5, 2]]))

    def test_apply_to_coords(self):
        grid = create_grid(4, 5)

        # .....
        # #....
        # .....
        # .....
        assert GridTransform.IDENTITY.apply_to_coords(grid, 0, 1) == (0, 1)

        # ..#.
        # ....
        # ....
        # ....
        # ....
        assert GridTransform.ROT_90.apply_to_coords(grid, 0, 1) == (2, 0)

        # .....
        # .....
        # ....#
        # .....
        assert GridTransform.ROT_180.apply_to_coords(grid, 0, 1) == (4, 2)

        # ....
        # ....
        # ....
        # ....
        # .#..
        assert GridTransform.ROT_270.apply_to_coords(grid, 0, 1) == (1, 4)

        # .....
        # ....#
        # .....
        # .....
        assert GridTransform.FLIP_H.apply_to_coords(grid, 0, 1) == (4, 1)

        # .....
        # .....
        # #....
        # .....
        assert GridTransform.FLIP_V.apply_to_coords(grid, 0, 1) == (0, 2)

        # .#..
        # ....
        # ....
        # ....
        # ....
        assert GridTransform.TRANSPOSE.apply_to_coords(grid, 0, 1) == (1, 0)

        # ....
        # ....
        # ....
        # ....
        # ..#.
        assert GridTransform.TRANSPOSE_ALT.apply_to_coords(grid, 0, 1) == (2, 4)


class TestTransformBasic:
    def test_default(self):
        scene = render_scene(SerifScene.Config(), (4, 5))
        assert_raw_grid(
            # we have to use outer grid instead of assert_grid(scene, ...) because the grid is transformed
            scene.area.outer_grid,
            """
                .....
                #....
                .....
                .....
            """,
        )

    def test_rot90(self):
        scene = render_scene(SerifScene.Config(transform=GridTransform.ROT_90), (4, 5))
        assert_raw_grid(
            scene.area.outer_grid,
            """
                ...#.
                .....
                .....
                .....
            """,
        )

    def test_rot180(self):
        scene = render_scene(SerifScene.Config(transform=GridTransform.ROT_180), (4, 5))
        assert_raw_grid(
            scene.area.outer_grid,
            """
                .....
                .....
                ....#
                .....
            """,
        )

    def test_rot270(self):
        scene = render_scene(SerifScene.Config(transform=GridTransform.ROT_270), (4, 5))
        assert_raw_grid(
            scene.area.outer_grid,
            """
                .....
                .....
                .....
                .#...
            """,
        )

    def test_flip_h(self):
        scene = render_scene(SerifScene.Config(transform=GridTransform.FLIP_H), (4, 5))
        assert_raw_grid(
            scene.area.outer_grid,
            """
                .....
                ....#
                .....
                .....
            """,
        )

    def test_flip_v(self):
        scene = render_scene(SerifScene.Config(transform=GridTransform.FLIP_V), (4, 5))
        assert_raw_grid(
            scene.area.outer_grid,
            """
                .....
                .....
                #....
                .....
            """,
        )

    def test_transpose_alt(self):
        scene = render_scene(SerifScene.Config(transform=GridTransform.TRANSPOSE), (4, 5))
        assert_raw_grid(
            scene.area.outer_grid,
            """
                .#...
                .....
                .....
                .....
            """,
        )

    def test_transpose(self):
        scene = render_scene(SerifScene.Config(transform=GridTransform.TRANSPOSE_ALT), (4, 5))
        assert_raw_grid(
            scene.area.outer_grid,
            """
                .....
                .....
                .....
                ...#.
            """,
        )


class TestSubarea:
    def test_area(self):
        scene = render_scene(SerifScene.Config(transform=GridTransform.ROT_90), (4, 5))
        assert scene.area.width == 5
        assert scene.area.height == 4
        child_area = scene.select_areas(AreaQuery())[0]
        assert child_area.width == 3
        assert child_area.height == 2

    def test_nested_transforms(self):
        scene = render_scene(
            SerifScene.Config(
                transform=GridTransform.ROT_90,
                children=[
                    ChildrenAction(
                        scene=SerifScene.Config(transform=GridTransform.ROT_90), where=AreaWhere(tags=["inner"])
                    )
                ],
            ),
            (6, 7),
        )
        assert_raw_grid(
            scene.area.outer_grid,
            """
                .....#.
                .......
                .......
                .....#.
                .......
                .......
            """,
        )
