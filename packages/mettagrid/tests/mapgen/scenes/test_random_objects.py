from mettagrid.mapgen.random.float import FloatUniformDistribution
from mettagrid.mapgen.scenes.random_objects import RandomObjects
from mettagrid.test_support.mapgen import render_scene


def test_objects():
    scene = render_scene(
        RandomObjects.Config(object_ranges={"altar": FloatUniformDistribution(low=0.2, high=0.5)}),
        (10, 10),
    )

    altar_count = (scene.grid == "altar").sum()
    assert 0.2 * 100 <= altar_count <= 0.5 * 100
