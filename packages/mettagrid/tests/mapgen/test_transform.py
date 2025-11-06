import numpy as np

import mettagrid.map_builder.utils
import mettagrid.mapgen.area
import mettagrid.mapgen.scene
import mettagrid.test_support.mapgen


class SerifConfig(mettagrid.mapgen.scene.SceneConfig):
    pass


class SerifScene(mettagrid.mapgen.scene.Scene[SerifConfig]):
    def render(self):
        self.grid[1, 0] = "wall"
        self.make_area(1, 1, self.width - 2, self.height - 2, tags=["inner"])


class TestGridTransformClass:
    def test_identity(self):
        assert (
            mettagrid.mapgen.scene.GridTransform.IDENTITY.compose(mettagrid.mapgen.scene.GridTransform.IDENTITY)
            == mettagrid.mapgen.scene.GridTransform.IDENTITY
        )
        assert (
            mettagrid.mapgen.scene.GridTransform.IDENTITY.compose(mettagrid.mapgen.scene.GridTransform.ROT_90)
            == mettagrid.mapgen.scene.GridTransform.ROT_90
        )

    def test_rot_composition(self):
        assert (
            mettagrid.mapgen.scene.GridTransform.ROT_90.compose(mettagrid.mapgen.scene.GridTransform.ROT_90)
            == mettagrid.mapgen.scene.GridTransform.ROT_180
        )
        assert (
            mettagrid.mapgen.scene.GridTransform.ROT_90.compose(mettagrid.mapgen.scene.GridTransform.ROT_180)
            == mettagrid.mapgen.scene.GridTransform.ROT_270
        )
        assert (
            mettagrid.mapgen.scene.GridTransform.ROT_90.compose(mettagrid.mapgen.scene.GridTransform.ROT_270)
            == mettagrid.mapgen.scene.GridTransform.IDENTITY
        )

    def test_inverse(self):
        assert mettagrid.mapgen.scene.GridTransform.IDENTITY.inverse() == mettagrid.mapgen.scene.GridTransform.IDENTITY
        assert mettagrid.mapgen.scene.GridTransform.ROT_90.inverse() == mettagrid.mapgen.scene.GridTransform.ROT_270
        assert mettagrid.mapgen.scene.GridTransform.ROT_180.inverse() == mettagrid.mapgen.scene.GridTransform.ROT_180
        assert mettagrid.mapgen.scene.GridTransform.ROT_270.inverse() == mettagrid.mapgen.scene.GridTransform.ROT_90
        assert mettagrid.mapgen.scene.GridTransform.FLIP_H.inverse() == mettagrid.mapgen.scene.GridTransform.FLIP_H
        assert mettagrid.mapgen.scene.GridTransform.FLIP_V.inverse() == mettagrid.mapgen.scene.GridTransform.FLIP_V
        assert (
            mettagrid.mapgen.scene.GridTransform.TRANSPOSE_ALT.inverse()
            == mettagrid.mapgen.scene.GridTransform.TRANSPOSE_ALT
        )
        assert (
            mettagrid.mapgen.scene.GridTransform.TRANSPOSE.inverse() == mettagrid.mapgen.scene.GridTransform.TRANSPOSE
        )

    def test_apply(self):
        grid = np.array([[0, 1, 2], [3, 4, 5]])
        np.testing.assert_array_equal(
            mettagrid.mapgen.scene.GridTransform.ROT_90.apply(grid), np.array([[3, 0], [4, 1], [5, 2]])
        )

    def test_apply_to_coords(self):
        grid = mettagrid.map_builder.utils.create_grid(4, 5)

        # .....
        # #....
        # .....
        # .....
        assert mettagrid.mapgen.scene.GridTransform.IDENTITY.apply_to_coords(grid, 0, 1) == (0, 1)

        # ..#.
        # ....
        # ....
        # ....
        # ....
        assert mettagrid.mapgen.scene.GridTransform.ROT_90.apply_to_coords(grid, 0, 1) == (2, 0)

        # .....
        # .....
        # ....#
        # .....
        assert mettagrid.mapgen.scene.GridTransform.ROT_180.apply_to_coords(grid, 0, 1) == (4, 2)

        # ....
        # ....
        # ....
        # ....
        # .#..
        assert mettagrid.mapgen.scene.GridTransform.ROT_270.apply_to_coords(grid, 0, 1) == (1, 4)

        # .....
        # ....#
        # .....
        # .....
        assert mettagrid.mapgen.scene.GridTransform.FLIP_H.apply_to_coords(grid, 0, 1) == (4, 1)

        # .....
        # .....
        # #....
        # .....
        assert mettagrid.mapgen.scene.GridTransform.FLIP_V.apply_to_coords(grid, 0, 1) == (0, 2)

        # .#..
        # ....
        # ....
        # ....
        # ....
        assert mettagrid.mapgen.scene.GridTransform.TRANSPOSE.apply_to_coords(grid, 0, 1) == (1, 0)

        # ....
        # ....
        # ....
        # ....
        # ..#.
        assert mettagrid.mapgen.scene.GridTransform.TRANSPOSE_ALT.apply_to_coords(grid, 0, 1) == (2, 4)


class TestTransformBasic:
    def test_default(self):
        scene = mettagrid.test_support.mapgen.render_scene(SerifScene.Config(), (4, 5))
        mettagrid.test_support.mapgen.assert_raw_grid(
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
        scene = mettagrid.test_support.mapgen.render_scene(
            SerifScene.Config(transform=mettagrid.mapgen.scene.GridTransform.ROT_90), (4, 5)
        )
        mettagrid.test_support.mapgen.assert_raw_grid(
            scene.area.outer_grid,
            """
                ...#.
                .....
                .....
                .....
            """,
        )

    def test_rot180(self):
        scene = mettagrid.test_support.mapgen.render_scene(
            SerifScene.Config(transform=mettagrid.mapgen.scene.GridTransform.ROT_180), (4, 5)
        )
        mettagrid.test_support.mapgen.assert_raw_grid(
            scene.area.outer_grid,
            """
                .....
                .....
                ....#
                .....
            """,
        )

    def test_rot270(self):
        scene = mettagrid.test_support.mapgen.render_scene(
            SerifScene.Config(transform=mettagrid.mapgen.scene.GridTransform.ROT_270), (4, 5)
        )
        mettagrid.test_support.mapgen.assert_raw_grid(
            scene.area.outer_grid,
            """
                .....
                .....
                .....
                .#...
            """,
        )

    def test_flip_h(self):
        scene = mettagrid.test_support.mapgen.render_scene(
            SerifScene.Config(transform=mettagrid.mapgen.scene.GridTransform.FLIP_H), (4, 5)
        )
        mettagrid.test_support.mapgen.assert_raw_grid(
            scene.area.outer_grid,
            """
                .....
                ....#
                .....
                .....
            """,
        )

    def test_flip_v(self):
        scene = mettagrid.test_support.mapgen.render_scene(
            SerifScene.Config(transform=mettagrid.mapgen.scene.GridTransform.FLIP_V), (4, 5)
        )
        mettagrid.test_support.mapgen.assert_raw_grid(
            scene.area.outer_grid,
            """
                .....
                .....
                #....
                .....
            """,
        )

    def test_transpose_alt(self):
        scene = mettagrid.test_support.mapgen.render_scene(
            SerifScene.Config(transform=mettagrid.mapgen.scene.GridTransform.TRANSPOSE), (4, 5)
        )
        mettagrid.test_support.mapgen.assert_raw_grid(
            scene.area.outer_grid,
            """
                .#...
                .....
                .....
                .....
            """,
        )

    def test_transpose(self):
        scene = mettagrid.test_support.mapgen.render_scene(
            SerifScene.Config(transform=mettagrid.mapgen.scene.GridTransform.TRANSPOSE_ALT), (4, 5)
        )
        mettagrid.test_support.mapgen.assert_raw_grid(
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
        scene = mettagrid.test_support.mapgen.render_scene(
            SerifScene.Config(transform=mettagrid.mapgen.scene.GridTransform.ROT_90), (4, 5)
        )
        assert scene.area.width == 5
        assert scene.area.height == 4
        child_area = scene.select_areas(mettagrid.mapgen.area.AreaQuery())[0]
        assert child_area.width == 3
        assert child_area.height == 2

    def test_nested_transforms(self):
        scene = mettagrid.test_support.mapgen.render_scene(
            SerifScene.Config(
                transform=mettagrid.mapgen.scene.GridTransform.ROT_90,
                children=[
                    mettagrid.mapgen.scene.ChildrenAction(
                        scene=SerifScene.Config(transform=mettagrid.mapgen.scene.GridTransform.ROT_90),
                        where=mettagrid.mapgen.area.AreaWhere(tags=["inner"]),
                    )
                ],
            ),
            (6, 7),
        )
        mettagrid.test_support.mapgen.assert_raw_grid(
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
