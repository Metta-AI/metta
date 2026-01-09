from mettagrid.mapgen.scenes.random import Random
from mettagrid.test_support.mapgen import render_scene


def test_objects():
    # Use 5x5 grid to ensure objects reliably fit
    scene = render_scene(
        Random.Config(objects={"assembler": 3, "temple": 2}),
        (5, 5),
    )

    assert (scene.grid == "assembler").sum() == 3
    assert (scene.grid == "temple").sum() == 2


def test_agents():
    scene = render_scene(Random.Config(agents=2), (3, 3))

    assert (scene.grid == "agent.agent").sum() == 2


def test_agents_dict():
    scene = render_scene(
        Random.Config(agents={"prey": 2, "predator": 1}),
        (3, 3),
    )

    assert (scene.grid == "agent.prey").sum() == 2
    assert (scene.grid == "agent.predator").sum() == 1
