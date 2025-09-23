import pytest
from pydantic import Field

from metta.common.tool import Tool
from metta.common.tool.run_tool import nestify, parse_cli_args
from mettagrid.config import Config


class InnerConfig(Config):
    value: int = 1


class DemoTool(Tool):
    name: str = "default"
    enabled: bool = False
    inner: InnerConfig = Field(default_factory=InnerConfig)

    def invoke(self, args: dict[str, str]) -> int | None:
        return 0


def test_parse_cli_args_supports_commander_formats() -> None:
    parsed = parse_cli_args(
        [
            "--inner.value",
            "-1",
            "--enabled",
            "--name=worker",
            "--payload",
            '{"foo": 1}',
            "threshold:0.5",
            "--ratio:2",
            "outer.depth=9",
        ]
    )

    assert parsed["inner.value"] == -1
    assert parsed["enabled"] is True
    assert parsed["name"] == "worker"
    assert parsed["payload"] == {"foo": 1}
    assert parsed["threshold"] == 0.5
    assert parsed["ratio"] == 2
    assert parsed["outer.depth"] == 9


def test_parse_cli_args_parses_strings_and_nulls() -> None:
    parsed = parse_cli_args(["--message", '"hello\\nworld"', "--maybe", "null"])

    assert parsed["message"] == "hello\nworld"
    assert parsed["maybe"] is None


def test_parse_cli_args_handles_dash_separator() -> None:
    parsed = parse_cli_args(["--flag", "--", "trailing=3"])

    assert parsed["flag"] is True
    assert parsed["trailing"] == 3


def test_parse_cli_args_requires_valid_format() -> None:
    with pytest.raises(ValueError):
        parse_cli_args(["invalid-token"])


def test_nestify_builds_nested_dicts() -> None:
    flat = {"inner.value": 7, "name": "worker"}
    assert nestify(flat) == {"inner": {"value": 7}, "name": "worker"}


def test_tool_model_validation_with_commander_args() -> None:
    cli_args = parse_cli_args(["--inner.value", "5", "--name=scenario", "--enabled"])
    payload = nestify(cli_args)

    tool = DemoTool.model_validate(payload)

    assert tool.name == "scenario"
    assert tool.enabled is True
    assert tool.inner.value == 5
