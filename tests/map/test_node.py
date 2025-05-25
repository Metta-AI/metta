import numpy as np
import pytest

from metta.map.node import Node

# Set a global seed for reproducibility
SEED = 42


class MockScene:
    def render(self, node):
        pass


@pytest.fixture
def node():
    # Set NumPy seed for reproducibility
    np.random.seed(SEED)

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


# Original tests remain unchanged
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
    assert np.random.get_state()[1][0] == SEED  # Verify seed is still effective
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


# === BENCHMARK TESTS ===


# Helper function for multiple iterations
def run_multiple_iterations(func, iterations=10):
    """
    Run a function multiple times and return the last result.
    This helps reduce variance in benchmarks.

    Args:
        func: Function to run (no arguments)
        iterations: Number of times to run the function

    Returns:
        The result of the last function call
    """
    # Reset seed for consistency
    np.random.seed(SEED)

    result = None
    for _ in range(iterations):
        result = func()
    return result


def test_benchmark_select_by_tag(benchmark, node):
    """Benchmark selecting areas by tag."""

    def select_by_tag():
        query = {"where": {"tags": ["tag2"]}}
        return node.select_areas(query)

    # Function that runs multiple iterations
    def run_select_by_tag():
        return run_multiple_iterations(select_by_tag, iterations=10)

    # Use the benchmark fixture directly
    results = benchmark(run_select_by_tag)

    # Verify it worked correctly
    assert len(results) == 2
    assert all("tag2" in area.tags for area in results)


def test_benchmark_select_with_limit_and_order(benchmark, node):
    """Benchmark selecting areas with limit and ordering."""

    def select_with_limit_order():
        query = {"limit": 2, "order_by": "first"}
        return node.select_areas(query)

    # Function that runs multiple iterations
    def run_select_with_limit_order():
        return run_multiple_iterations(select_with_limit_order, iterations=10)

    # Use the benchmark fixture directly
    results = benchmark(run_select_with_limit_order)

    # Verify it worked correctly
    assert len(results) == 2
    assert results[0].id == 0
    assert results[1].id == 1
