from metta.map.scenes.random_scene_from_dir import RandomSceneFromDir
from tests.map.scenes.utils import render_node


def test_basic(monkeypatch):
    node = render_node(
        RandomSceneFromDir,
        dict(dir="scenes/test"),
        (10, 10),
    )

    assert (node.grid == "wall").sum() > 0
