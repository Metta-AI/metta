from metta.map.scenes.random_scene_from_dir import RandomSceneFromDir
from tests.map.scenes.utils import render_scene


def test_basic(monkeypatch):
    scene = render_scene(
        RandomSceneFromDir,
        dict(dir="scenes/test"),
        (10, 10),
    )

    assert (scene.grid == "wall").sum() > 0
