from metta.map.scenes.random_scene_from_dir import RandomSceneFromDir
from tests.map.scenes.utils import scene_to_node


def test_basic(monkeypatch):
    scene = RandomSceneFromDir(dir="scenes/test")
    node = scene_to_node(scene, (10, 10))

    assert (node.grid == "wall").sum() > 0
