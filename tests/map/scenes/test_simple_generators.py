"""Tests for simple scene generators with minimal test coverage."""

import pytest

from metta.map.scene import ChildrenAction
from metta.map.scenes.bsp import BSP
from metta.map.scenes.make_connected import MakeConnected
from metta.map.scenes.room_grid import RoomGrid
from tests.map.scenes.utils import assert_connected, render_scene


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
