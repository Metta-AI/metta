import os

import mettagrid.mapgen.scenes.yaml
import mettagrid.test_support.mapgen


def test_yaml():
    file = f"{os.path.dirname(__file__)}/fixtures/test.yaml"
    scene = mettagrid.test_support.mapgen.render_scene(mettagrid.mapgen.scenes.yaml.YamlScene.Config(file=file), (3, 3))

    assert (scene.grid == "altar").sum() == 3
    assert (scene.grid == "temple").sum() == 2
