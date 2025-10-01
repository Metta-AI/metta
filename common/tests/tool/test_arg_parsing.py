"""Unit tests for argument parsing and classification utilities."""

from pydantic import Field

from metta.common.tool import Tool
from metta.common.tool.run_tool import (
    classify_remaining_args,
    get_tool_fields,
    nestify,
    parse_cli_args,
    parse_value,
)
from mettagrid.base_config import Config


class NestedConfig(Config):
    field: str = "nested_default"
    another_field: int = 100


class SimpleTestTool(Tool):
    tool_name = "simple_test"
    value: str = "default"
    nested: NestedConfig = Field(default_factory=NestedConfig)

    def invoke(self, args: dict[str, str]) -> int | None:
        return 0


def test_parse_value_types():
    """Verify parse_value correctly identifies and converts different types."""
    # Booleans
    assert parse_value("true") is True
    assert parse_value("FALSE") is False

    # Integers
    assert parse_value("42") == 42
    assert parse_value("-100") == -100

    # Floats
    assert parse_value("3.14") == 3.14

    # Strings remain strings when not parseable
    assert parse_value("hello") == "hello"
    assert parse_value("42abc") == "42abc"
    assert parse_value("") == ""


def test_parse_value_json_structures():
    """Verify JSON objects and arrays are parsed correctly."""
    assert parse_value('{"k":1}') == {"k": 1}
    assert parse_value("[1,2,3]") == [1, 2, 3]

    # Invalid JSON falls back to string
    assert parse_value("{not json}") == "{not json}"

    # Oversized JSON-like content returns original string (security limit)
    big = "{" + "x" * 1_000_001 + "}"
    result = parse_value(big)
    assert isinstance(result, str)
    assert len(result) > 1_000_000


def test_parse_cli_args_basic():
    """Verify CLI arguments are parsed into a flat dict with correct types."""
    result = parse_cli_args(["name=test", "count=42", "enabled=true", "nested.field=value"])

    assert result["name"] == "test"
    assert result["count"] == 42
    assert result["enabled"] is True
    assert result["nested.field"] == "value"


def test_parse_cli_args_handles_special_cases():
    """Verify edge cases like empty values and embedded equals signs."""
    # Empty value
    result = parse_cli_args(["key="])
    assert result["key"] == ""

    # Value containing equals signs
    result = parse_cli_args(["url=http://example.com?a=1&b=2"])
    assert result["url"] == "http://example.com?a=1&b=2"


def test_parse_cli_args_rejects_invalid_format():
    """Verify that arguments without '=' are rejected."""
    import pytest

    with pytest.raises(ValueError, match="Invalid argument format"):
        parse_cli_args(["no_equals_sign"])

    with pytest.raises(ValueError, match="Invalid argument format"):
        parse_cli_args([""])


def test_nestify_converts_flat_to_nested():
    """Verify nestify converts dotted keys into nested dicts."""
    flat = {
        "a.b.c": 1,
        "a.b.d": 2,
        "a.e": 3,
        "x": 4,
    }

    nested = nestify(flat)

    assert nested["a"]["b"]["c"] == 1
    assert nested["a"]["b"]["d"] == 2
    assert nested["a"]["e"] == 3
    assert nested["x"] == 4


def test_get_tool_fields_includes_parent_fields():
    """Verify get_tool_fields returns fields from the tool and its parent classes."""
    fields = get_tool_fields(SimpleTestTool)

    # Tool's own fields
    assert "value" in fields
    assert "nested" in fields

    # Parent Tool class fields
    assert "system" in fields


def test_classify_remaining_args_separates_known_and_unknown():
    """Verify classification separates tool overrides from unknown arguments."""
    tool_fields = get_tool_fields(SimpleTestTool)

    remaining = {
        "value": "custom",
        "nested.field": "updated",
        "nested.another_field": 999,
        "system.device": "cpu",
        "unknown_top_level": "x",
        "unknown.nested.field": "y",
    }

    overrides, unknown = classify_remaining_args(remaining, tool_fields)

    # Known tool fields are classified as overrides
    assert overrides["value"] == "custom"
    assert overrides["nested.field"] == "updated"
    assert overrides["nested.another_field"] == 999
    assert overrides["system.device"] == "cpu"

    # Unknown fields are flagged
    assert "unknown_top_level" in unknown
    assert "unknown.nested.field" in unknown
