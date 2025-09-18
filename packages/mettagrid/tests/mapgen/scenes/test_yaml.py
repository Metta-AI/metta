import os

from mettagrid.mapgen.scenes.yaml import YamlScene
from mettagrid.test_support.mapgen import render_scene


def test_yaml():
    file = f"{os.path.dirname(__file__)}/fixtures/test.yaml"
    scene = render_scene(YamlScene.factory(YamlScene.Params(file=file)), (3, 3))

    assert (scene.grid == "altar").sum() == 3
    assert (scene.grid == "temple").sum() == 2
