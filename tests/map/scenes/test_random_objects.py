from metta.map.scenes.random_objects import RandomObjects
from tests.map.scenes.utils import scene_to_node


def test_objects():
    scene = RandomObjects(object_ranges={"mine": ("uniform", 0.3, 0.5)})
    node = scene_to_node(scene, (3, 3))

    mine_count = (node.grid == "mine").sum()
    assert 0.3 * 9 <= mine_count <= 0.5 * 9
