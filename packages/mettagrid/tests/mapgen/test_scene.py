import numpy as np
import pytest

from mettagrid.mapgen.area import Area, AreaQuery, AreaWhere
from mettagrid.mapgen.scene import ChildrenAction, Scene, SceneConfig


class MockConfig(SceneConfig):
    pass


class MockScene(Scene[MockConfig]):
    def render(self):
        pass


def make_scene(children_actions: list[ChildrenAction]):
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
    area = Area.root_area_from_grid(grid)
    scene = MockScene.Config(seed=42, children=children_actions).create_root(area=area)
    # Create some test areas with different tags
    scene.make_area(0, 0, 3, 2, tags=["tag1", "tag2", "scene1"])  # ABC / FGH
    scene.make_area(1, 2, 2, 2, tags=["tag2", "tag3", "scene2"])  # LM / QR
    scene.make_area(3, 2, 2, 3, tags=["tag1", "tag3", "scene3"])  # NO / ST / XY
    return scene


@pytest.fixture
def scene():
    return make_scene([])


def test_areas_are_correctly_created(scene):
    assert scene._areas[0].tags == ["tag1", "tag2", "scene1"]
    assert np.array_equal(scene._areas[0].grid, np.array([["A", "B", "C"], ["F", "G", "H"]]))
    assert scene._areas[1].tags == ["tag2", "tag3", "scene2"]
    assert np.array_equal(scene._areas[1].grid, np.array([["L", "M"], ["Q", "R"]]))
    assert scene._areas[2].tags == ["tag1", "tag3", "scene3"]
    assert np.array_equal(scene._areas[2].grid, np.array([["N", "O"], ["S", "T"], ["X", "Y"]]))


class TestSelectAreas:
    def test_where_tags(self, scene):
        # Test selecting areas with specific tags
        query = AreaQuery(where=AreaWhere(tags=["tag1", "tag2"]))
        selected_areas = scene.select_areas(query)
        assert len(selected_areas) == 1
        assert "scene1" in selected_areas[0].tags  # First area has both tags
        # Test selecting areas with single tag
        query = AreaQuery(where=AreaWhere(tags=["tag2"]))
        selected_areas = scene.select_areas(query)
        assert len(selected_areas) == 2  # Two areas have tag2
        assert all("tag2" in area.tags for area in selected_areas)

    def test_where_full(self, scene):
        # Test selecting the full area
        query = AreaQuery(where="full")
        selected_areas = scene.select_areas(query)
        assert len(selected_areas) == 1
        assert selected_areas[0] == scene.area

    def test_limit(self, scene):
        # Test limiting number of results
        query = AreaQuery(limit=2)
        selected_areas = scene.select_areas(query)
        assert len(selected_areas) == 2
        # Test with order_by="first"
        query = AreaQuery(limit=2, order_by="first")
        selected_areas = scene.select_areas(query)
        assert len(selected_areas) == 2
        assert "scene1" in selected_areas[0].tags
        assert "scene2" in selected_areas[1].tags
        # Test with order_by="last"
        query = AreaQuery(limit=2, order_by="last")
        selected_areas = scene.select_areas(query)
        assert len(selected_areas) == 2
        assert "scene2" in selected_areas[0].tags
        assert "scene3" in selected_areas[1].tags

    def test_lock(self, scene):
        # Test locking mechanism
        query = AreaQuery(lock="test_lock", order_by="first", limit=1)
        selected_areas = scene.select_areas(query)
        assert len(selected_areas) == 1
        assert "scene1" in selected_areas[0].tags
        # When we query again, we skip the locked area
        selected_areas2 = scene.select_areas(query)
        assert len(selected_areas2) == 1
        assert "scene2" in selected_areas2[0].tags

    def test_offset(self, scene):
        # Test offset with first ordering
        query = AreaQuery(limit=2, order_by="first", offset=1)
        selected_areas = scene.select_areas(query)
        assert len(selected_areas) == 2
        assert "scene2" in selected_areas[0].tags
        assert "scene3" in selected_areas[1].tags
        # Test offset with last ordering
        query = AreaQuery(limit=2, order_by="last", offset=1)
        selected_areas = scene.select_areas(query)
        assert len(selected_areas) == 2
        assert "scene1" in selected_areas[0].tags
        assert "scene2" in selected_areas[1].tags

    def test_returns_list_type(self, scene):
        """Test that select_areas always returns a list, not a numpy array"""
        # Test with no query
        selected_areas = scene.select_areas(AreaQuery())
        assert isinstance(selected_areas, list), "select_areas should return a list"

        # Test with random ordering (which uses numpy internally)
        query = AreaQuery(limit=2, order_by="random")
        selected_areas = scene.select_areas(query)
        assert isinstance(selected_areas, list), "select_areas with random ordering should return a list"

        # Test with first ordering
        query = AreaQuery(limit=2, order_by="first")
        selected_areas = scene.select_areas(query)
        assert isinstance(selected_areas, list), "select_areas with first ordering should return a list"

        # Test with last ordering
        query = AreaQuery(limit=2, order_by="last")
        selected_areas = scene.select_areas(query)
        assert isinstance(selected_areas, list), "select_areas with last ordering should return a list"

        # Verify list operations work
        query = AreaQuery(limit=1, order_by="random")
        selected_areas = scene.select_areas(query)
        # This should not raise AttributeError if it's a proper list
        selected_areas_copy = selected_areas.copy()
        assert len(selected_areas_copy) == 1


class TestSceneTree:
    def test_basic(self, scene):
        scene_tree = scene.get_scene_tree()
        assert scene_tree["config"]["type"] == "tests.mapgen.test_scene.MockScene"
        assert scene_tree["area"] == scene.area.as_dict()
        assert len(scene_tree["config"]["children"]) == 0

    def test_with_children(self):
        scene = make_scene(
            [
                ChildrenAction(
                    scene=MockScene.Config(),
                    where=AreaWhere(tags=["tag1"]),
                )
            ]
        )
        scene.render_with_children()
        scene_tree = scene.get_scene_tree()
        assert scene_tree["config"]["type"] == "tests.mapgen.test_scene.MockScene"
        assert scene_tree["area"] == scene.area.as_dict()
        assert len(scene_tree["children"]) == 2
        assert scene_tree["children"][0]["config"]["type"] == "tests.mapgen.test_scene.MockScene"
