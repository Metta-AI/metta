import mettagrid.mapgen.random.float
import mettagrid.mapgen.scenes.random_objects
import mettagrid.test_support.mapgen


def test_objects():
    scene = mettagrid.test_support.mapgen.render_scene(
        mettagrid.mapgen.scenes.random_objects.RandomObjects.Config(
            object_ranges={"altar": mettagrid.mapgen.random.float.FloatUniformDistribution(low=0.2, high=0.5)}
        ),
        (10, 10),
    )

    altar_count = (scene.grid == "altar").sum()
    assert 0.2 * 100 <= altar_count <= 0.5 * 100
