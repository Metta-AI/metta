import numpy as np
import pytest

from deps.mettagrid.mettagrid.map.node import Node


class MockScene:
    def render(self, node):
        pass


@pytest.fixture
def node():
    # Create a 5x5 grid with some test data
    grid = np.array(
        [
            ["A", "B", "C", "D", "E"],
            ["F", "G", "H", "I", "J"],
            ["K", "L", "M", "N", "O"],
            ["P", "Q", "R", "S", "T"],
            ["U", "V", "W", "X", "Y"],
        ]
    )
    node = Node(MockScene(), grid)

    # Create some test areas with different tags
    node.make_area(0, 0, 3, 2, tags=["tag1", "tag2"])  # ABC / FGH
    node.make_area(1, 2, 2, 2, tags=["tag2", "tag3"])  # LM / QR
    node.make_area(3, 2, 2, 3, tags=["tag1", "tag3"])  # NO / ST / XY

    return node


def test_areas_are_correctly_created(node):
    assert node._areas[0].id == 0
    assert node._areas[0].tags == ["tag1", "tag2"]
    assert np.array_equal(node._areas[0].grid, np.array([["A", "B", "C"], ["F", "G", "H"]]))
    assert node._areas[1].id == 1
    assert node._areas[1].tags == ["tag2", "tag3"]
    assert np.array_equal(node._areas[1].grid, np.array([["L", "M"], ["Q", "R"]]))
    assert node._areas[2].id == 2
    assert node._areas[2].tags == ["tag1", "tag3"]
    assert np.array_equal(node._areas[2].grid, np.array([["N", "O"], ["S", "T"], ["X", "Y"]]))


def test_select_areas_with_where_tags(node):
    # Test selecting areas with specific tags
    query = {"where": {"tags": ["tag1", "tag2"]}}
    selected_areas = node.select_areas(query)
    assert len(selected_areas) == 1
    assert selected_areas[0].id == 0  # First area has both tags

    # Test selecting areas with single tag
    query = {"where": {"tags": ["tag2"]}}
    selected_areas = node.select_areas(query)
    assert len(selected_areas) == 2  # Two areas have tag2
    assert all("tag2" in area.tags for area in selected_areas)


def test_select_areas_with_where_full(node):
    # Test selecting the full area
    query = {"where": "full"}
    selected_areas = node.select_areas(query)
    assert len(selected_areas) == 1
    assert selected_areas[0].id == -1  # Full area has id -1


def test_select_areas_with_limit(node):
    # Test limiting number of results
    query = {"limit": 2}
    selected_areas = node.select_areas(query)
    assert len(selected_areas) == 2

    # Test with order_by="first"
    query = {"limit": 2, "order_by": "first"}
    selected_areas = node.select_areas(query)
    assert len(selected_areas) == 2
    assert selected_areas[0].id == 0
    assert selected_areas[1].id == 1

    # Test with order_by="last"
    query = {"limit": 2, "order_by": "last"}
    selected_areas = node.select_areas(query)
    assert len(selected_areas) == 2
    assert selected_areas[0].id == 1
    assert selected_areas[1].id == 2


def test_select_areas_with_lock(node):
    # Test locking mechanism
    query = {"lock": "test_lock", "order_by": "first", "limit": 1}
    selected_areas = node.select_areas(query)
    assert len(selected_areas) == 1
    assert selected_areas[0].id == 0

    # When we query again, we skip the locked area
    selected_areas2 = node.select_areas(query)
    assert len(selected_areas2) == 1
    assert selected_areas2[0].id == 1


def test_select_areas_with_offset(node):
    # Test offset with first ordering
    query = {"limit": 2, "order_by": "first", "offset": 1}
    selected_areas = node.select_areas(query)
    assert len(selected_areas) == 2
    assert selected_areas[0].id == 1
    assert selected_areas[1].id == 2

    # Test offset with last ordering
    query = {"limit": 2, "order_by": "last", "offset": 1}
    selected_areas = node.select_areas(query)
    assert len(selected_areas) == 2
    assert selected_areas[0].id == 0
    assert selected_areas[1].id == 1
