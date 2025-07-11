#!/usr/bin/env python3
"""Test suite for the commander function."""

from tools.commander import CommanderError, commander


def test_basic_functionality():
    """Test basic flag parsing."""
    print("Test 1: Basic functionality")

    tree = {"foo": {"bar": {"baz": 1}}}

    # Test various flag formats
    result = commander("--foo.bar.baz 3", tree.copy())
    assert result["foo"]["bar"]["baz"] == 3
    print("  ✓ --foo.bar.baz 3")

    result = commander("-foo.bar.baz 3", tree.copy())
    assert result["foo"]["bar"]["baz"] == 3
    print("  ✓ -foo.bar.baz 3")

    result = commander("--foo.bar.baz=3", tree.copy())
    assert result["foo"]["bar"]["baz"] == 3
    print("  ✓ --foo.bar.baz=3")

    result = commander("--foo.bar.baz:3", tree.copy())
    assert result["foo"]["bar"]["baz"] == 3
    print("  ✓ --foo.bar.baz:3")


def test_string_handling():
    """Test string parsing with quotes and escapes."""
    print("\nTest 2: String handling")

    tree = {"message": "old", "special": "@bar"}

    # Quoted strings
    result = commander('--message "hello world"', tree.copy())
    assert result["message"] == "hello world"
    print('  ✓ --message "hello world"')

    result = commander("--message 'hello world'", tree.copy())
    assert result["message"] == "hello world"
    print("  ✓ --message 'hello world'")

    # Multi-line strings
    result = commander('--message "hello\nworld"', tree.copy())
    assert result["message"] == "hello\nworld"
    print('  ✓ --message "hello\\nworld"')

    # Escaped quotes
    result = commander('--message "hello \\"world\\""', tree.copy())
    assert result["message"] == 'hello "world"'
    print('  ✓ --message "hello \\"world\\""')

    # Unquoted strings
    result = commander("--message hello", tree.copy())
    assert result["message"] == "hello"
    print("  ✓ --message hello")

    # Special characters (must be quoted)
    result = commander("--special '@foo'", tree.copy())
    assert result["special"] == "@foo"
    print("  ✓ --special '@foo'")


def test_boolean_flags():
    """Test boolean flag handling."""
    print("\nTest 3: Boolean flags")

    tree = {"verbose": False, "debug": False}

    # Single boolean flag
    result = commander("--verbose", tree.copy())
    assert result["verbose"] is True
    print("  ✓ --verbose")

    # Multiple boolean flags
    result = commander("--verbose --debug", tree.copy())
    assert result["verbose"] is True
    assert result["debug"] is True
    print("  ✓ --verbose --debug")


def test_numbers():
    """Test number parsing."""
    print("\nTest 4: Numbers")

    tree = {"count": 0, "rate": 0.0}

    # Integers
    result = commander("--count 42", tree.copy())
    assert result["count"] == 42
    print("  ✓ --count 42")

    # Negative numbers
    result = commander("--count -1", tree.copy())
    assert result["count"] == -1
    print("  ✓ --count -1")

    # Positive numbers with +
    result = commander("--count +1", tree.copy())
    assert result["count"] == 1
    print("  ✓ --count +1")

    # Floats
    result = commander("--rate 1.23456789", tree.copy())
    assert result["rate"] == 1.23456789
    print("  ✓ --rate 1.23456789")

    # Scientific notation
    result = commander("--rate 1.23456789e10", tree.copy())
    assert result["rate"] == 1.23456789e10
    print("  ✓ --rate 1.23456789e10")


def test_arrays():
    """Test array indexing."""
    print("\nTest 5: Arrays")

    tree = {"array": [1, 2, 3]}

    # Modify array element
    result = commander("--array.1 7", tree.copy())
    assert result["array"] == [1, 7, 3]
    print("  ✓ --array.1 7")


def test_type_checking():
    """Test type checking."""
    print("\nTest 6: Type checking")

    tree = {"count": 42}

    # Should reject string for int
    try:
        commander('--count "hello"', tree.copy())
        raise AssertionError("Should have failed type check")
    except CommanderError:
        print("  ✓ Type mismatch detected correctly")


def test_error_handling():
    """Test error conditions."""
    print("\nTest 7: Error handling")

    tree = {"foo": {"bar": 1}}

    # Missing key
    try:
        commander("--foo.missing 42", tree.copy())
        raise AssertionError("Should have failed on missing key")
    except CommanderError:
        print("  ✓ Missing key detected correctly")


def test_help():
    """Test help functionality."""
    print("\nTest 8: Help (output suppressed)")

    tree = {"foo": {"bar": 1}, "verbose": False}

    # This should print help and return tree unchanged
    import io
    import sys

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        result = commander("--help", tree.copy())
        assert result == tree
        print("  ✓ --help works")
    finally:
        sys.stdout = old_stdout


def test_complex_scenarios():
    """Test complex scenarios."""
    print("\nTest 9: Complex scenarios")

    tree = {
        "config": {"timeout": 30, "enabled": False, "items": [1, 2, 3]},
        "metadata": {"version": "1.0", "build": None},
    }

    # Multiple arguments
    result = commander("--config.timeout 60 --config.enabled", tree.copy())
    assert result["config"]["timeout"] == 60
    assert result["config"]["enabled"] is True
    print("  ✓ Multiple arguments")

    # Replace array element
    result = commander("--config.items.2 99", tree.copy())
    assert result["config"]["items"] == [1, 2, 99]
    print("  ✓ Array replacement")

    # Null handling
    result = commander("--metadata.build null", tree.copy())
    assert result["metadata"]["build"] is None
    print("  ✓ Null handling")

    # Negative number parsing
    result = commander("--config.timeout -5", tree.copy())
    assert result["config"]["timeout"] == -5
    print("  ✓ Negative number handling")

    # List of arguments (sys.argv style)
    result = commander(["--config.timeout", "120", "--config.enabled"], tree.copy())
    assert result["config"]["timeout"] == 120
    assert result["config"]["enabled"] is True
    print("  ✓ List of arguments (sys.argv style)")


# Test classes for object support
class Baz:
    def __init__(self):
        self.value = 42


class Bar:
    def __init__(self):
        self.count = 10
        self.enabled = False
        self.baz = Baz()


class Foo:
    def __init__(self):
        self.message = "hello"
        self.bar = Bar()
        self.items = [1, 2, 3]


def test_python_objects():
    """Test Python object support."""
    print("\nTest 10: Python objects")

    tree = Foo()

    # Modify object attributes
    result = commander("--bar.count 25", tree)
    assert result.bar.count == 25
    print("  ✓ --bar.count 25")

    # Nested object modification
    result = commander("--bar.baz.value 100", tree)
    assert result.bar.baz.value == 100
    print("  ✓ --bar.baz.value 100")

    # String modification
    result = commander('--message "updated message"', tree)
    assert result.message == "updated message"
    print('  ✓ --message "updated message"')

    # Boolean flag
    result = commander("--bar.enabled", tree)
    assert result.bar.enabled is True
    print("  ✓ --bar.enabled")

    # Array in object
    result = commander("--items.1 99", tree)
    assert result.items == [1, 99, 3]
    print("  ✓ --items.1 99")

    # Multiple modifications
    tree2 = Foo()
    result = commander("--bar.count 50 --bar.enabled --message test", tree2)
    assert result.bar.count == 50
    assert result.bar.enabled is True
    assert result.message == "test"
    print("  ✓ Multiple object modifications")

    # Help for objects
    import io
    import sys

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        tree3 = Foo()
        result = commander("--help", tree3)
        assert result == tree3
        print("  ✓ Object help works")
    finally:
        sys.stdout = old_stdout

    # Type mismatch on objects
    try:
        tree4 = Foo()
        commander('--bar.count "invalid"', tree4)
        raise AssertionError("Should have failed type check")
    except CommanderError:
        print("  ✓ Object type mismatch detected correctly")

    # Missing attribute
    try:
        tree5 = Foo()
        commander("--bar.missing 42", tree5)
        raise AssertionError("Should have failed on missing attribute")
    except CommanderError:
        print("  ✓ Object missing attribute detected correctly")


if __name__ == "__main__":
    print("=== Commander Test Suite ===\n")

    test_basic_functionality()
    test_string_handling()
    test_boolean_flags()
    test_numbers()
    test_arrays()
    test_type_checking()
    test_error_handling()
    test_help()
    test_complex_scenarios()
    test_python_objects()

    print("\n✅ All tests passed!")
