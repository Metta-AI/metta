"""Fast unit tests for parse_value/parse_cli_args/get_tool_fields/classify_remaining_args."""

from pydantic import Field

from metta.common.tool import Tool
from metta.common.tool.run_tool import (
    classify_remaining_args,
    get_tool_fields,
    parse_cli_args,
    parse_value,
)
from mettagrid.config import Config


class NestedConfig(Config):
    field: str = "nested_default"
    another_field: int = 100


class SimpleTestTool(Tool):
    value: str = "default"
    nested: NestedConfig = Field(default_factory=NestedConfig)

    def invoke(self, args: dict[str, str]) -> int | None:
        return 0


def test_parse_value():
    # booleans
    assert parse_value("true") is True
    assert parse_value("FALSE") is False
    # ints
    assert parse_value("42") == 42
    assert parse_value("-100") == -100
    # floats
    assert parse_value("3.14") == 3.14
    # strings
    assert parse_value("hello") == "hello"
    assert parse_value("42abc") == "42abc"
    assert parse_value("") == ""
    # json
    assert parse_value('{"k":1}') == {"k": 1}
    assert parse_value("[1,2]") == [1, 2]
    # invalid json returns original
    assert parse_value("{not json}") == "{not json}"
    # oversized json-like returns original
    big = "{" + "a" * 1_000_001 + "}"
    assert parse_value(big) == big


def test_parse_cli_args():
    args = ["key1=value1", "key2=123", "nested.field=test", "flag=true"]
    result = parse_cli_args(args)
    assert result == {"key1": "value1", "key2": 123, "nested.field": "test", "flag": True}


def test_get_tool_fields():
    fields = get_tool_fields(SimpleTestTool)
    assert "value" in fields
    assert "nested" in fields
    assert "system" in fields  # from Tool base


def test_classify_remaining_args_for_tool_fields():
    tool_fields = get_tool_fields(SimpleTestTool)
    remaining = {
        "value": "override_value",
        "nested.field": "custom",
        "nested.another_field": 999,
        "system.device": "cuda",
        "unknown": "x",
        "deeply.nested.field": "y",
    }
    overrides, unknown = classify_remaining_args(remaining, tool_fields)
    assert overrides == {
        "value": "override_value",
        "nested.field": "custom",
        "nested.another_field": 999,
        "system.device": "cuda",
    }
    assert unknown == ["unknown", "deeply.nested.field"]


def test_parse_cli_args_error_cases():
    import pytest

    with pytest.raises(ValueError, match="Expected key=value format"):
        parse_cli_args(["invalid_arg"])
    with pytest.raises(ValueError, match="Expected key=value format"):
        parse_cli_args([""])
    with pytest.raises(ValueError, match="non-empty key"):
        parse_cli_args(["=value"])


def test_parse_cli_args_edge_cases():
    # empty value
    assert parse_cli_args(["key="]) == {"key": ""}
    # value with '='
    assert parse_cli_args(["k=v=with=equals"]) == {"k": "v=with=equals"}
    # dotted keys stay flat
    assert parse_cli_args(["a.b.c=1", "a.b.d=2", "a.e=3"]) == {"a.b.c": 1, "a.b.d": 2, "a.e": 3}
