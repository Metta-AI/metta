from metta.map.scenes.random_objects import RandomObjects
from tests.map.scenes.utils import scene_to_node


def test_objects():
    scene = RandomObjects(object_ranges={"mine": ("uniform", 0.2, 0.5)})
    node = scene_to_node(scene, (10, 10))

    mine_count = (node.grid == "mine").sum()
    assert 0.2 * 100 <= mine_count <= 0.5 * 100
