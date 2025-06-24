import logging
import sys

import pytest

from metta.common.memory import get_object_size, log_object_memory


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


def test_logging_output(caplog):
    """Test that logging produces correct output."""
    test_data = {"key": [1, 2, 3] * 100}

    # Test with name
    with caplog.at_level(logging.INFO):
        log_object_memory(test_data, "test_data", logging.INFO)

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO
    assert "Memory usage for 'test_data':" in caplog.text
    assert "MB" in caplog.text
    assert "bytes" in caplog.text

    # Test without name
    caplog.clear()
    with caplog.at_level(logging.INFO):
        log_object_memory(test_data)

    assert f"Memory usage for 'Object_{id(test_data)}':" in caplog.text

    # Test different log level
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        log_object_memory(test_data, "debug_test", logging.DEBUG)

    assert caplog.records[0].levelno == logging.DEBUG


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
