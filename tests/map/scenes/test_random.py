from metta.map.scenes.random import Random
from tests.map.scenes.utils import scene_to_node


def test_objects():
    scene = Random(objects={"mine": 3, "generator": 2})
    node = scene_to_node(scene, (3, 3))

    assert (node.grid == "mine").sum() == 3
    assert (node.grid == "generator").sum() == 2


def test_agents():
    scene = Random(agents=2)
    node = scene_to_node(scene, (3, 3))

    assert (node.grid == "agent.agent").sum() == 2


def test_agents_dict():
    scene = Random(agents={"prey": 2, "predator": 1})
    node = scene_to_node(scene, (3, 3))

    assert (node.grid == "agent.prey").sum() == 2
    assert (node.grid == "agent.predator").sum() == 1
