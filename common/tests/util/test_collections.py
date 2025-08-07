"""Tests for metta.common.util.collections module."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pytest

from metta.common.util.collections import group_by, remove_none_values


class TestGroupBy:
    """Test cases for the group_by function."""

    def test_group_by_empty_list(self):
        """Test group_by with an empty list."""
        result = group_by([], lambda x: x)
        assert isinstance(result, defaultdict)
        assert len(result) == 0
        assert list(result.keys()) == []

    def test_group_by_single_item(self):
        """Test group_by with a single item list."""
        result = group_by([5], lambda x: x % 2)
        assert len(result) == 1
        assert result[1] == [5]  # 5 % 2 = 1

    def test_group_by_integers_by_even_odd(self):
        """Test group_by with integers grouped by even/odd."""
        numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = group_by(numbers, lambda x: x % 2)

        assert len(result) == 2
        assert set(result.keys()) == {0, 1}  # 0 for even, 1 for odd
        assert result[0] == [2, 4, 6, 8, 10]  # even numbers
        assert result[1] == [1, 3, 5, 7, 9]   # odd numbers

    def test_group_by_strings_by_length(self):
        """Test group_by with strings grouped by length."""
        words = ["cat", "dog", "elephant", "a", "bird", "ox"]
        result = group_by(words, len)

        assert len(result) == 5
        assert result[1] == ["a"]
        assert result[2] == ["ox"]
        assert result[3] == ["cat", "dog"]
        assert result[4] == ["bird"]
        assert result[8] == ["elephant"]

    def test_group_by_strings_by_first_letter(self):
        """Test group_by with strings grouped by first letter."""
        words = ["apple", "banana", "cherry", "apricot", "blueberry", "avocado"]
        result = group_by(words, lambda word: word[0])

        assert len(result) == 3
        assert set(result.keys()) == {"a", "b", "c"}
        assert result["a"] == ["apple", "apricot", "avocado"]
        assert result["b"] == ["banana", "blueberry"]
        assert result["c"] == ["cherry"]

    def test_group_by_with_dataclass(self):
        """Test group_by with custom objects (dataclass)."""
        @dataclass
        class Person:
            name: str
            age: int
            department: str

        people = [
            Person("Alice", 30, "Engineering"),
            Person("Bob", 25, "Marketing"),
            Person("Charlie", 35, "Engineering"),
            Person("Diana", 28, "Marketing"),
            Person("Eve", 32, "Sales"),
        ]

        # Group by department
        result = group_by(people, lambda p: p.department)

        assert len(result) == 3
        assert set(result.keys()) == {"Engineering", "Marketing", "Sales"}

        engineering = result["Engineering"]
        assert len(engineering) == 2
        assert {p.name for p in engineering} == {"Alice", "Charlie"}

        marketing = result["Marketing"]
        assert len(marketing) == 2
        assert {p.name for p in marketing} == {"Bob", "Diana"}

        sales = result["Sales"]
        assert len(sales) == 1
        assert sales[0].name == "Eve"



    def test_group_by_with_complex_key_function(self):
        """Test group_by with complex key extraction."""
        data = [
            {"user": "alice", "score": 85, "category": "A"},
            {"user": "bob", "score": 92, "category": "B"},
            {"user": "charlie", "score": 78, "category": "A"},
            {"user": "diana", "score": 95, "category": "A"},
            {"user": "eve", "score": 88, "category": "B"},
        ]

        # Group by category
        result = group_by(data, lambda item: item["category"])

        assert len(result) == 2
        assert len(result["A"]) == 3
        assert len(result["B"]) == 2

        # Check users in category A
        category_a_users = {item["user"] for item in result["A"]}
        assert category_a_users == {"alice", "charlie", "diana"}

    def test_group_by_maintains_order(self):
        """Test that group_by maintains order within groups."""
        items = ["first", "second", "third", "fourth", "fifth"]
        result = group_by(items, lambda word: len(word))

        # Items with same length should maintain relative order
        assert result[5] == ["first", "third", "fifth"]  # 5-letter words in order
        assert result[6] == ["second", "fourth"]         # 6-letter words in order

    def test_group_by_returns_defaultdict(self):
        """Test that group_by returns a defaultdict that creates empty lists for missing keys."""
        result = group_by([1, 2, 3], lambda x: x % 2)

        # Access a key that doesn't exist
        missing_key = result[99]
        assert missing_key == []
        assert isinstance(missing_key, list)

        # The defaultdict should now have this key
        assert 99 in result


class TestRemoveNoneValues:
    """Test cases for the remove_none_values function."""

    def test_remove_none_values_empty_dict(self):
        """Test remove_none_values with an empty dictionary."""
        result = remove_none_values({})
        assert result == {}
        assert isinstance(result, dict)

    def test_remove_none_values_no_none_values(self):
        """Test remove_none_values with no None values."""
        original = {"a": 1, "b": "hello", "c": [1, 2, 3], "d": {"nested": "dict"}}
        result = remove_none_values(original)

        assert result == original
        # Should be a new dict, not the same instance
        assert result is not original

    def test_remove_none_values_all_none_values(self):
        """Test remove_none_values with all None values."""
        original = {"a": None, "b": None, "c": None}
        result = remove_none_values(original)

        assert result == {}

    def test_remove_none_values_mixed_values(self):
        """Test remove_none_values with mixed None and non-None values."""
        original = {
            "keep1": "value",
            "remove1": None,
            "keep2": 42,
            "remove2": None,
            "keep3": [],
            "remove3": None,
            "keep4": False,  # False should be kept, not removed
            "keep5": 0,      # 0 should be kept, not removed
            "keep6": "",     # Empty string should be kept, not removed
        }

        expected = {
            "keep1": "value",
            "keep2": 42,
            "keep3": [],
            "keep4": False,
            "keep5": 0,
            "keep6": "",
        }

        result = remove_none_values(original)
        assert result == expected

    def test_remove_none_values_preserves_types(self):
        """Test that remove_none_values preserves the types of non-None values."""
        original = {
            "string": "text",
            "int": 123,
            "float": 45.67,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "bool_true": True,
            "bool_false": False,
            "none_value": None,
        }

        result = remove_none_values(original)

        # None value should be removed
        assert "none_value" not in result

        # All other values should be preserved with correct types
        assert result["string"] == "text"
        assert isinstance(result["string"], str)

        assert result["int"] == 123
        assert isinstance(result["int"], int)

        assert result["float"] == 45.67
        assert isinstance(result["float"], float)

        assert result["list"] == [1, 2, 3]
        assert isinstance(result["list"], list)

        assert result["dict"] == {"nested": "value"}
        assert isinstance(result["dict"], dict)

        assert result["bool_true"] is True
        assert isinstance(result["bool_true"], bool)

        assert result["bool_false"] is False
        assert isinstance(result["bool_false"], bool)



    def test_remove_none_values_real_world_use_case(self):
        """Test remove_none_values with a real-world HTTP headers scenario."""
        # Simulating HTTP headers where some might be None
        headers = {
            "Authorization": "Bearer token123",
            "Content-Type": "application/json",
            "X-Custom-Header": "custom-value",
            "X-Optional-Header": None,
            "User-Agent": "MyApp/1.0",
            "X-Auth-Token": None,  # This is what we saw in the actual usage
        }

        result = remove_none_values(headers)

        expected = {
            "Authorization": "Bearer token123",
            "Content-Type": "application/json",
            "X-Custom-Header": "custom-value",
            "User-Agent": "MyApp/1.0",
        }

        assert result == expected
        # Ensure None headers are completely removed
        assert "X-Optional-Header" not in result
        assert "X-Auth-Token" not in result

    def test_remove_none_values_type_annotations_work(self):
        """Test that the function works correctly with type annotations."""
        # The function signature is: dict[K, T | None] -> dict[K, T]
        # This tests that the typing works as expected

        input_dict: dict[str, int | None] = {
            "a": 1,
            "b": None,
            "c": 3,
        }

        result: dict[str, int] = remove_none_values(input_dict)

        assert result == {"a": 1, "c": 3}
        # All values in result should be non-None
        for value in result.values():
            assert value is not None
