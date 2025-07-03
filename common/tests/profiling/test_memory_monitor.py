import sys

from metta.common.profiling.memory_monitor import get_object_size


def test_basic_memory_calculation():
    """Test that memory size calculation works for basic types."""
    # Simple types
    assert get_object_size(42) == sys.getsizeof(42)
    assert get_object_size("hello") == sys.getsizeof("hello")

    # List with items
    test_list = [1, 2, 3, 4, 5]
    expected = sys.getsizeof(test_list) + sum(sys.getsizeof(i) for i in test_list)
    assert get_object_size(test_list) == expected

    # Dict with items
    test_dict = {"a": 1, "b": 2}
    expected = sys.getsizeof(test_dict) + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in test_dict.items())
    assert get_object_size(test_dict) == expected


def test_object_with_attributes():
    """Test memory calculation for objects with __dict__."""

    class TestObj:
        def __init__(self):
            self.x = 100
            self.y = "test"

    obj = TestObj()
    size = get_object_size(obj)

    # Should include object + __dict__ + attributes
    min_expected = sys.getsizeof(obj) + sys.getsizeof(obj.__dict__)
    assert size > min_expected


def test_memory_growth():
    """Test that memory grows with data size."""
    small = list(range(10))
    large = list(range(10000))

    assert get_object_size(small) < get_object_size(large)
    assert get_object_size(large) > 100000  # Should be > 100KB
