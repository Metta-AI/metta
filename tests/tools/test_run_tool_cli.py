import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic import Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_local_module(name: str, relative_path: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


# Load dependencies from local workspace before importing the runner module.
metta_common_tool_module = load_local_module("metta.common.tool", "common/src/metta/common/tool/__init__.py")
mettagrid_config_module = load_local_module(
    "mettagrid.config.config", "packages/mettagrid/python/src/mettagrid/config/config.py"
)
run_tool = load_local_module("metta.common.tool.run_tool", "common/src/metta/common/tool/run_tool.py")

# ruff: noqa: E402
Tool = metta_common_tool_module.Tool
Config = mettagrid_config_module.Config
_normalize_tokens = run_tool._normalize_tokens
nestify = run_tool.nestify
parse_cli_args = run_tool.parse_cli_args


class InnerConfig(Config):
    value: int = 1


class DemoTool(Tool):
    name: str = "default"
    enabled: bool = False
    inner: InnerConfig = Field(default_factory=InnerConfig)

    def invoke(self, args: dict[str, str]) -> int | None:
        return 0


def test_parse_cli_args_supports_commander_formats() -> None:
    raw_args = [
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

    canonical = _normalize_tokens(raw_args)
    parsed = parse_cli_args(canonical)

    assert parsed["inner.value"] == -1
    assert parsed["enabled"] is True
    assert parsed["name"] == "worker"
    assert parsed["payload"] == {"foo": 1}
    assert parsed["threshold"] == 0.5
    assert parsed["ratio"] == 2
    assert parsed["outer.depth"] == 9


def test_parse_cli_args_parses_strings_and_nulls() -> None:
    canonical = _normalize_tokens(["--message", '"hello\\nworld"', "--maybe", "null"])
    parsed = parse_cli_args(canonical)

    assert parsed["message"] == "hello\nworld"
    assert parsed["maybe"] is None


def test_parse_cli_args_handles_dash_separator() -> None:
    canonical = _normalize_tokens(["--flag", "--", "trailing=3"])
    parsed = parse_cli_args(canonical)

    assert parsed["flag"] is True
    assert parsed["trailing"] == 3


def test_parse_cli_args_requires_valid_format() -> None:
    with pytest.raises(ValueError):
        parse_cli_args(["invalid-token"])


def test_nestify_builds_nested_dicts() -> None:
    flat = {"inner.value": 7, "name": "worker"}
    assert nestify(flat) == {"inner": {"value": 7}, "name": "worker"}


def test_tool_model_validation_with_commander_args() -> None:
    canonical = _normalize_tokens(["--inner.value", "5", "--name=scenario", "--enabled"])
    cli_args = parse_cli_args(canonical)
    payload = nestify(cli_args)

    tool = DemoTool.model_validate(payload)

    assert tool.name == "scenario"
    assert tool.enabled is True
    assert tool.inner.value == 5
