import os

import mettagrid.mapgen.scenes.random_yaml_scene
import mettagrid.test_support.mapgen


def test_objects():
    # Same as test_random_scene.py, but with YAML files
    w_count = 0
    a_count = 0

    dir = f"{os.path.dirname(__file__)}/fixtures/dir1"

    # 1 / 2^30 chance of failure
    for _ in range(30):
        scene = mettagrid.test_support.mapgen.render_scene(
            mettagrid.mapgen.scenes.random_yaml_scene.RandomYamlScene.Config(
                candidates=[
                    mettagrid.mapgen.scenes.random_yaml_scene.RandomYamlSceneCandidate(scene_file=f"{dir}/1.yaml"),
                    mettagrid.mapgen.scenes.random_yaml_scene.RandomYamlSceneCandidate(scene_file=f"{dir}/2.yaml"),
                ]
            ),
            (1, 1),
        )
        w_count += (scene.grid == "wall").sum()
        a_count += (scene.grid == "altar").sum()

    assert w_count > 0
    assert a_count > 0
