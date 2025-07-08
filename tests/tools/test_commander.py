#!/usr/bin/env python3
"""Tests for the commander function."""

import contextlib
import copy
import io

from tools.commander import commander

print("=== Commander Test Suite ===\n")

# Test 1: Basic functionality
print("Test 1: Basic functionality")
tree = {"foo": {"bar": {"baz": 0}, "name": "test"}, "array": [1, 2, 3], "flag": False}

test_cases = [
    ("--foo.bar.baz 3", {"foo": {"bar": {"baz": 3}, "name": "test"}, "array": [1, 2, 3], "flag": False}),
    ("-foo.bar.baz 3", {"foo": {"bar": {"baz": 3}, "name": "test"}, "array": [1, 2, 3], "flag": False}),
    ("--foo.bar.baz=3", {"foo": {"bar": {"baz": 3}, "name": "test"}, "array": [1, 2, 3], "flag": False}),
    ("--foo.bar.baz:3", {"foo": {"bar": {"baz": 3}, "name": "test"}, "array": [1, 2, 3], "flag": False}),
]

for args, expected in test_cases:
    tree_copy = copy.deepcopy(tree)
    result = commander(args, tree_copy)
    assert result == expected, f"Failed: {args}\nExpected: {expected}\nGot: {result}"
    print(f"  ✓ {args}")

# Test 2: String handling
print("\nTest 2: String handling")
tree = {"message": "", "special": ""}

test_cases = [
    ('--message "hello world"', {"message": "hello world", "special": ""}),
    ("--message 'hello world'", {"message": "hello world", "special": ""}),
    ('--message "hello\nworld"', {"message": "hello\nworld", "special": ""}),
    ('--message "hello \\"world\\""', {"message": 'hello "world"', "special": ""}),
    ("--message hello", {"message": "hello", "special": ""}),
    ("--special '@foo'", {"message": "", "special": "@foo"}),
]

for args, expected in test_cases:
    tree_copy = copy.deepcopy(tree)
    result = commander(args, tree_copy)
    assert result == expected, f"Failed: {args}\nExpected: {expected}\nGot: {result}"
    print(f"  ✓ {args}")

# Test 3: Boolean flags
print("\nTest 3: Boolean flags")
tree = {"verbose": False, "debug": False}

result = commander("--verbose", copy.deepcopy(tree))
assert result["verbose"]
print("  ✓ --verbose")

result = commander("--verbose --debug", copy.deepcopy(tree))
assert result["verbose"] and result["debug"]
print("  ✓ --verbose --debug")

# Test 4: Numbers
print("\nTest 4: Numbers")
tree = {"count": 0, "rate": 0.0, "offset": 0}

test_cases = [
    ("--count 42", {"count": 42, "rate": 0.0, "offset": 0}),
    ("--count -1", {"count": -1, "rate": 0.0, "offset": 0}),
    ("--count +1", {"count": 1, "rate": 0.0, "offset": 0}),
    ("--rate 1.23456789", {"count": 0, "rate": 1.23456789, "offset": 0}),
    ("--rate 1.23456789e10", {"count": 0, "rate": 1.23456789e10, "offset": 0}),
]

for args, expected in test_cases:
    tree_copy = copy.deepcopy(tree)
    result = commander(args, tree_copy)
    assert result == expected, f"Failed: {args}\nExpected: {expected}\nGot: {result}"
    print(f"  ✓ {args}")

# Test 5: Arrays
print("\nTest 5: Arrays")
tree = {"array": [0, 0, 0]}

result = commander("--array.1 7", copy.deepcopy(tree))
assert result["array"] == [0, 7, 0]
print("  ✓ --array.1 7")

# Test 6: JSON5 values
print("\nTest 6: JSON5 values")
tree = {"config": {}, "items": []}

test_cases = [
    ('--config {name: "test", value: 3}', {"config": {"name": "test", "value": 3}, "items": []}),
    ("--items [1, 2, 3]", {"config": {}, "items": [1, 2, 3]}),
    ("--config {a: 1, b: 2,}", {"config": {"a": 1, "b": 2}, "items": []}),  # trailing comma
    ("--config {'x': 1}", {"config": {"x": 1}, "items": []}),  # single quotes
    ('--config {nested: {deep: "value"}}', {"config": {"nested": {"deep": "value"}}, "items": []}),  # nested
]

for args, expected in test_cases:
    tree_copy = copy.deepcopy(tree)
    result = commander(args, tree_copy)
    assert result == expected, f"Failed: {args}\nExpected: {expected}\nGot: {result}"
    print(f"  ✓ {args}")

# Test 7: Type checking
print("\nTest 7: Type checking")
tree = {"string": "hello", "number": 42}

try:
    commander("--string 123", copy.deepcopy(tree))
    raise AssertionError("Should have raised TypeError")
except ValueError as e:
    assert "Type mismatch" in str(e)
    print("  ✓ Type mismatch detected correctly")

# Test 8: Error handling
print("\nTest 8: Error handling")
tree = {"foo": {"bar": 1}}

try:
    commander("--foo.baz 1", copy.deepcopy(tree))
    raise AssertionError("Should have raised KeyError")
except ValueError as e:
    assert "Key 'baz' not found" in str(e)
    print("  ✓ Missing key detected correctly")

# Test 9: Help
print("\nTest 9: Help (output suppressed)")
f = io.StringIO()
with contextlib.redirect_stdout(f):
    commander("--help", tree)
help_output = f.getvalue()
assert "--foo.bar:" in help_output
print("  ✓ --help works")

# Test 10: Complex scenarios
print("\nTest 10: Complex scenarios")
tree = {
    "server": {"host": "localhost", "port": 8080, "ssl": False},
    "database": {"connections": 10, "timeout": 30.5},
    "features": ["auth", "api"],
    "metadata": None,
}

# Multiple arguments
result = commander("--server.port 9000 --server.ssl --database.connections 20", copy.deepcopy(tree))
assert result["server"]["port"] == 9000
assert result["server"]["ssl"]
assert result["database"]["connections"] == 20
print("  ✓ Multiple arguments")

# Complex JSON5 with nested arrays and objects
result = commander('--features ["auth", "api", "admin"]', copy.deepcopy(tree))
assert result["features"] == ["auth", "api", "admin"]
print("  ✓ Array replacement")

# Null/None handling
result = commander("--metadata null", copy.deepcopy(tree))
assert result["metadata"] is None
print("  ✓ Null handling")

# Edge case: negative number that looks like a flag
offset_tree = {"offset": 0}
result = commander("--offset -42", copy.deepcopy(offset_tree))
assert result["offset"] == -42
print("  ✓ Negative number handling")

# Test list of arguments (sys.argv style)
args_list = ["--server.port", "3000", "--server.ssl"]
result = commander(args_list, copy.deepcopy(tree))
assert result["server"]["port"] == 3000
assert result["server"]["ssl"]
print("  ✓ List of arguments (sys.argv style)")

print("\n✅ All tests passed!")
