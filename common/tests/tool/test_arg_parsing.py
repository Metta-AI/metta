"""Fast unit tests for arg parsing logic without subprocess overhead."""

from typing import Any

from pydantic import Field

from metta.common.tool import Tool
from metta.common.tool.run_tool import (
    classify_remaining_args,
    extract_function_args,
    get_tool_fields,
    parse_cli_args,
    parse_value,
)
from metta.mettagrid.config import Config


class NestedConfig(Config):
    """A nested configuration object for testing."""

    field: str = "nested_default"
    another_field: int = 100


class SimpleTestTool(Tool):
    """A simple tool with fields for testing."""

    value: str = "default"
    nested: NestedConfig = Field(default_factory=NestedConfig)

    def invoke(self, args: dict[str, str]) -> int | None:
        return 0


def make_test_tool(run: str = "default_run", count: int = 42) -> SimpleTestTool:
    """Function that creates a test tool."""
    return SimpleTestTool()


def split_args_and_overrides_test_helper(
    cli_args: dict[str, Any], make_tool_cfg: Any, tool_cfg: Tool
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """
    Test helper that mimics the old split_args_and_overrides behavior.
    Uses the two separate helper functions from run_tool.
    """
    # Phase 1: Extract function args
    func_args, remaining = extract_function_args(cli_args, make_tool_cfg)

    # Phase 2: Classify remaining as overrides or unknown
    tool_fields = get_tool_fields(type(tool_cfg))
    overrides, unknown = classify_remaining_args(remaining, tool_fields)

    return func_args, overrides, unknown


class TestArgParsing:
    """Unit tests for argument parsing and classification logic."""

    def test_parse_value(self):
        """Test value parsing for different types."""
        # Boolean values
        assert parse_value("true") is True
        assert parse_value("True") is True
        assert parse_value("TRUE") is True
        assert parse_value("false") is False
        assert parse_value("False") is False
        assert parse_value("FALSE") is False

        # Integer values
        assert parse_value("42") == 42
        assert parse_value("-100") == -100
        assert parse_value("0") == 0

        # Float values
        assert parse_value("3.14") == 3.14
        assert parse_value("-2.5") == -2.5
        assert parse_value("0.0") == 0.0

        # String values (including those that look numeric but aren't)
        assert parse_value("hello") == "hello"
        assert parse_value("42abc") == "42abc"
        assert parse_value("3.14.15") == "3.14.15"
        assert parse_value("") == ""

        # JSON values
        assert parse_value('{"key": "value"}') == {"key": "value"}
        assert parse_value("[1, 2, 3]") == [1, 2, 3]
        assert parse_value('{"nested": {"key": 123}}') == {"nested": {"key": 123}}

        # Invalid JSON that looks like JSON (should return as string)
        assert parse_value("{not valid json}") == "{not valid json}"
        assert parse_value('[missing quote: "test]') == '[missing quote: "test]'

        # Very large JSON-like string (should return as string due to size limit)
        large_json_like = "{" + "a" * 1_000_001 + "}"
        assert parse_value(large_json_like) == large_json_like

    def test_parse_cli_args(self):
        """Test CLI argument parsing."""
        args = ["key1=value1", "key2=123", "nested.field=test"]
        result = parse_cli_args(args)

        assert result == {"key1": "value1", "key2": 123, "nested.field": "test"}

    def test_get_tool_fields(self):
        """Test getting fields from Tool class."""
        fields = get_tool_fields(SimpleTestTool)

        # Should include fields from SimpleTestTool and parent Tool class
        assert "value" in fields
        assert "nested" in fields
        assert "system" in fields  # From parent Tool class

    def test_split_args_and_overrides_function(self):
        """Test splitting args for a function that returns a Tool."""
        cli_args = {
            "run": "test_run",
            "count": 100,
            "value": "override_value",
            "nested.field": "custom",
            "unknown_arg": "test",
        }

        temp_tool = make_test_tool()
        func_args, overrides, unknown = split_args_and_overrides_test_helper(cli_args, make_test_tool, temp_tool)

        # Check function args
        assert func_args == {"run": "test_run", "count": 100}

        # Check overrides
        assert overrides == {"value": "override_value", "nested.field": "custom"}

        # Check unknown
        assert unknown == ["unknown_arg"]

    def test_split_args_and_overrides_class(self):
        """Test splitting args for a Tool class constructor."""
        cli_args = {"value": "test_value", "system.device": "cpu", "unknown": "arg"}

        temp_tool = SimpleTestTool()
        func_args, overrides, unknown = split_args_and_overrides_test_helper(cli_args, SimpleTestTool, temp_tool)

        # Tool constructor takes no args (besides self)
        assert func_args == {}

        # All valid fields should be overrides
        assert overrides == {"value": "test_value", "system.device": "cpu"}

        # Unknown args
        assert unknown == ["unknown"]

    def test_nested_field_detection(self):
        """Test that nested fields are properly detected as overrides."""
        cli_args = {
            "nested.field": "test",
            "nested.another_field": 999,
            "system.device": "cuda",
            "deeply.nested.field": "value",
        }

        temp_tool = SimpleTestTool()
        func_args, overrides, unknown = split_args_and_overrides_test_helper(cli_args, make_test_tool, temp_tool)

        # No function args in this case
        assert func_args == {}

        # Known nested fields should be treated as overrides
        assert overrides == {"nested.field": "test", "nested.another_field": 999, "system.device": "cuda"}

        # Unknown base field "deeply" should make this unknown
        assert unknown == ["deeply.nested.field"]

    def test_parse_cli_args_error_cases(self):
        """Test CLI argument parsing error cases."""
        import pytest

        # Invalid format (no equals sign)
        with pytest.raises(ValueError, match="Invalid argument format"):
            parse_cli_args(["invalid_arg"])

        # Empty argument
        with pytest.raises(ValueError, match="Invalid argument format"):
            parse_cli_args([""])

    def test_parse_cli_args_edge_cases(self):
        """Test CLI argument parsing edge cases."""
        # Empty value
        result = parse_cli_args(["key="])
        assert result == {"key": ""}

        # Value with equals sign
        result = parse_cli_args(["key=value=with=equals"])
        assert result == {"key": "value=with=equals"}

        # Dotted keys are kept flat
        result = parse_cli_args(["a.b.c=1", "a.b.d=2", "a.e=3"])
        assert result == {"a.b.c": 1, "a.b.d": 2, "a.e": 3}

    def test_integration_scenario(self):
        """Test a complete integration scenario."""
        # Simulate what happens when the tool is called
        cli_args_raw = ["run=my_test", "count=99", "value=custom", "nested.field=updated", "system.device=cpu"]

        # Parse the arguments
        cli_args = parse_cli_args(cli_args_raw)

        # Create temp tool to inspect fields
        temp_tool = make_test_tool()

        # Split into function args and overrides
        func_args, overrides, unknown = split_args_and_overrides_test_helper(cli_args, make_test_tool, temp_tool)

        # Verify the split
        assert func_args == {"run": "my_test", "count": 99}
        # Now parse_cli_args keeps dotted keys flat
        assert overrides == {"value": "custom", "nested.field": "updated", "system.device": "cpu"}
        assert unknown == []

        # Create the actual tool with function args
        tool = make_test_tool(**func_args)

        # Apply overrides (this would normally be done by tool.override())
        assert tool.value == "default"  # Before override
        assert tool.nested.field == "nested_default"  # Before override

    def test_function_args_have_precedence(self):
        """Test that function args take precedence over tool fields in case of conflicts."""

        # Create a function that has parameters that conflict with tool fields
        def conflicting_function(
            value: str = "func_default",  # Conflicts with SimpleTestTool.value
            system: str = "func_system",  # Conflicts with Tool.system
            new_param: str = "test",
        ) -> SimpleTestTool:
            return SimpleTestTool()

        cli_args = {
            "value": "test_value",  # Could be function arg OR tool field
            "system": "test_system",  # Could be function arg OR tool field
            "new_param": "test_param",  # Only function arg
            "nested": "test_nested",  # Only tool field
        }

        temp_tool = SimpleTestTool()
        func_args, overrides, unknown = split_args_and_overrides_test_helper(cli_args, conflicting_function, temp_tool)

        # Function parameters should take precedence
        assert func_args == {
            "value": "test_value",  # Function arg wins
            "system": "test_system",  # Function arg wins
            "new_param": "test_param",  # Function arg
        }

        # Only non-conflicting tool fields should be overrides
        assert overrides == {
            "nested": "test_nested"  # Tool field (no conflict)
        }

        assert unknown == []
