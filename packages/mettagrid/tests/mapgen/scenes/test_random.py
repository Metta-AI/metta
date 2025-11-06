import mettagrid.mapgen.scenes.random
import mettagrid.test_support.mapgen


def test_objects():
    scene = mettagrid.test_support.mapgen.render_scene(
        mettagrid.mapgen.scenes.random.Random.Config(objects={"altar": 3, "temple": 2}),
        (3, 3),
    )

    assert (scene.grid == "altar").sum() == 3
    assert (scene.grid == "temple").sum() == 2


def test_agents():
    scene = mettagrid.test_support.mapgen.render_scene(mettagrid.mapgen.scenes.random.Random.Config(agents=2), (3, 3))

    assert (scene.grid == "agent.agent").sum() == 2


def test_agents_dict():
    scene = mettagrid.test_support.mapgen.render_scene(
        mettagrid.mapgen.scenes.random.Random.Config(agents={"prey": 2, "predator": 1}),
        (3, 3),
    )

    assert (scene.grid == "agent.prey").sum() == 2
    assert (scene.grid == "agent.predator").sum() == 1
