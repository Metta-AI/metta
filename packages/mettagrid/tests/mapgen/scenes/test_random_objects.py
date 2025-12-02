from mettagrid.mapgen.random.float import FloatUniformDistribution
from mettagrid.mapgen.scenes.random_objects import RandomObjects
from mettagrid.test_support.mapgen import render_scene


def test_objects():
    scene = render_scene(
        RandomObjects.Config(object_ranges={"assembler": FloatUniformDistribution(low=0.2, high=0.5)}),
        (10, 10),
    )

    assembler_count = (scene.grid == "assembler").sum()
    assert 0.2 * 100 <= assembler_count <= 0.5 * 100
