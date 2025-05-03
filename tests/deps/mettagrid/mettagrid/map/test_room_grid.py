import numpy as np
import pytest

from mettagrid.map.node import Node
from mettagrid.map.scenes.room_grid import RoomGrid


class MockScene:
    def render(self, node):
        pass


@pytest.fixture
def node():
    # Create a 10x10 grid for testing
    grid = np.full((10, 10), "empty", dtype="<U50")
    return Node(MockScene(), grid)


def test_room_grid_with_rows_columns(node):
    # Test creating a 2x3 grid of rooms
    scene = RoomGrid(rows=2, columns=3, border_width=1, border_object="wall")
    scene.render(node)
    # Verify the grid structure
    # Should have walls at inner borders
    assert np.array_equal(node.grid[4, :], ["wall"] * 10)  # Horizontal border
    assert np.array_equal(node.grid[:, 2], ["wall"] * 10)  # Vertical border
    areas = node.select_areas({})
    # Verify room areas are created. The 4x2 shape is due to the border width.
    assert len(areas) == 6
    assert all(area.grid.shape == (4, 2) for area in areas)
    # Verify room positions
    room_positions = [(area.grid[0, 0], area.grid[-1, -1]) for area in areas]
    assert all(pos[0] == "empty" and pos[1] == "empty" for pos in room_positions)


def test_room_grid_with_layout(node):
    # Test creating rooms with a specific layout and tags
    layout = [["room1", "room2"], ["room3", "room4"]]
    scene = RoomGrid(layout=layout, border_width=1, border_object="wall")
    scene.render(node)
    areas = node.select_areas({})
    # Verify room areas are created with correct tags
    assert len(areas) == 4
    assert all(area.grid.shape == (4, 4) for area in areas)
    # Verifying that the tagged areas are where we expect them to be is a pain,
    # so for now just verify that the tags are what we expect.
    room_tags = [area.tags[0] for area in areas]
    assert set(room_tags) == {"room1", "room2", "room3", "room4"}


# === BENCHMARK TESTS ===


def test_benchmark_room_grid_creation_small(benchmark):
    """Benchmark creating a small room grid (3x3)."""

    def create_small_grid():
        # Create a new node with a larger grid to fit the rooms
        grid = np.full((100, 100), "empty", dtype="<U50")
        test_node = Node(MockScene(), grid)

        scene = RoomGrid(rows=3, columns=3, border_width=1, border_object="wall")
        scene.render(test_node)
        return test_node.select_areas({})

    # Run the benchmark
    areas = benchmark(create_small_grid)

    # Verify it worked correctly
    assert len(areas) == 9  # 3x3 grid should have 9 rooms


def test_benchmark_room_grid_creation_large(benchmark):
    """Benchmark creating a large room grid (10x10)."""

    def create_large_grid():
        # Create a new node with a larger grid to fit the rooms
        grid = np.full((100, 100), "empty", dtype="<U50")
        test_node = Node(MockScene(), grid)

        scene = RoomGrid(rows=10, columns=10, border_width=1, border_object="wall")
        scene.render(test_node)
        return test_node.select_areas({})

    # Run the benchmark
    areas = benchmark(create_large_grid)

    # Verify it worked correctly
    assert len(areas) == 100  # 10x10 grid should have 100 rooms


def test_benchmark_area_selection_by_tag(benchmark, node):
    """Benchmark selecting areas by tag in a room grid."""
    # Setup: Create a grid with tagged rooms
    layout = [["kitchen", "living_room", "bathroom"], ["bedroom", "hall", "office"], ["storage", "garage", "patio"]]
    node.grid = np.full((30, 30), "empty", dtype="<U50")
    scene = RoomGrid(layout=layout, border_width=1, border_object="wall")
    scene.render(node)

    def select_areas_by_tag():
        # Specifically select areas with tags "living_room" or "bedroom"
        areas = []
        for area in node.select_areas({}):
            if area.tags and area.tags[0] in ["living_room", "bedroom"]:
                areas.append(area)
        return areas

    # Run the benchmark
    areas = benchmark(select_areas_by_tag)

    # Verify results
    assert len(areas) == 2
    assert all(area.tags[0] in ["living_room", "bedroom"] for area in areas)
