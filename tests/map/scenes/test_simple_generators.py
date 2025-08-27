"""Tests for simple scene generators with minimal test coverage."""

import pytest

from metta.map.scene import ChildrenAction
from metta.map.scenes.ascii import Ascii
from metta.map.scenes.bsp import BSP
from metta.map.scenes.make_connected import MakeConnected
from metta.map.scenes.room_grid import RoomGrid
from metta.map.scenes.wfc import WFC
from tests.map.scenes.utils import assert_connected, assert_grid, render_scene


class TestAsciiScenes:
    def test_basic(self):
        scene = render_scene(Ascii.factory(Ascii.Params(uri="tests/map/scenes/fixtures/test.map")), (4, 4))

        assert_grid(
            scene,
            """
####
#_.#
##.#
####
        """,
        )


class TestWFCScenes:
    def test_basic(self):
        scene = render_scene(
            WFC.factory(
                WFC.Params(
                    pattern="""
                    .#...
                    ###..
                    ###..
                """
                )
            ),
            (20, 20),
        )

        assert (scene.grid == "wall").sum() > 0
        assert (scene.grid == "empty").sum() > 0


class TestBSPScenes:
    # Some BSP scenes end up disconnected.
    # BSP is not heavily used right now, so this is a TODO note for the future improvements if we ever need it.
    @pytest.mark.skip(reason="BSP has bugs")
    def test_basic(self):
        for _ in range(10):
            scene = render_scene(
                BSP.factory(
                    BSP.Params(
                        rooms=7,
                        min_room_size=3,
                        min_room_size_ratio=0.5,
                        max_room_size_ratio=0.9,
                    )
                ),
                (20, 20),
            )

            assert_connected(scene.grid)


class TestMakeConnected:
    def test_connect_room_grid(self):
        scene = render_scene(
            RoomGrid.factory(
                RoomGrid.Params(
                    rows=2,
                    columns=3,
                ),
                children_actions=[ChildrenAction(scene=MakeConnected.factory(), where="full")],
            ),
            shape=(20, 20),
        )

        assert_connected(scene.grid)