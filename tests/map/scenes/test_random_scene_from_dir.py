from pathlib import Path

from tests.map.scenes.utils import scene_to_node


def test_basic(monkeypatch):
    monkeypatch.setattr("metta.map.config.scenes_root", Path("tests/map/scenes/fixtures/scenes"))
    from metta.map.scenes.random_scene_from_dir import RandomSceneFromDir

    scene = RandomSceneFromDir(dir="tests/map/scenes/fixtures/scenes")
    node = scene_to_node(scene, (10, 10))

    assert (node.grid == "wall").sum() > 0
