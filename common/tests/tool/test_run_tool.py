import os
import subprocess
import sys

import pytest


@pytest.fixture
def with_extra_imports_root(monkeypatch):
    extra_imports_root = os.path.join(os.path.dirname(__file__), "fixtures/extra-import-root")
    monkeypatch.setenv("PYTHONPATH", extra_imports_root)


def run_tool(*args: str) -> subprocess.CompletedProcess[str]:
    """Helper to invoke the runner with text I/O, capturing both streams."""
    return subprocess.run(
        [sys.executable, "-m", "metta.common.tool.run_tool", *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_basic(with_extra_imports_root):
    # Behavior: the tool runs and prints from invoke()
    result = subprocess.check_output(
        [sys.executable, "-m", "metta.common.tool.run_tool", "mypackage.tools.TestTool"], text=True
    )
    assert "TestTool invoked" in result


def test_unknown_tool(with_extra_imports_root):
    # Behavior: unknown symbol yields non-zero with helpful error content
    result = run_tool("mypackage.tools.NoSuchTool")
    assert result.returncode > 0
    combined_output = result.stderr + result.stdout
    assert "has no" in combined_output and "attribute" in combined_output and "NoSuchTool" in combined_output


def test_unknown_module(with_extra_imports_root):
    # Behavior: unknown module yields non-zero with ImportError details
    result = run_tool("mypackage.no_such_tools.TestTool")
    assert result.returncode > 0
    combined_output = result.stderr + result.stdout
    assert "No module named" in combined_output and "mypackage.no_such_tools" in combined_output


def test_required_field_supported_for_tool_class(with_extra_imports_root):
    """
    Behavior: required Pydantic field on a Tool subclass is validated at construction.
    Edge case from review: `x=123` must be accepted and printed.
    """
    result = run_tool("mypackage.tools.RequiredFieldTool", "x=123")
    assert result.returncode == 0
    combined_output = result.stdout + result.stderr
    # The tool prints its x value from invoke()
    assert "\n123\n" in combined_output or " 123" in combined_output or "123" in combined_output


def test_dotted_overrides_are_nested_and_validated(with_extra_imports_root):
    """
    Behavior: dotted keys are interpreted as nested config and validated at construction time.
    SimpleTestTool prints nested fields; assert they reflect our CLI.
    """
    result = run_tool(
        "mypackage.tools.SimpleTestTool",
        "value=custom",
        "nested.field=updated",
        "nested.another_field=999",
    )
    assert result.returncode == 0
    out = result.stdout + result.stderr
    # Printed by SimpleTestTool.invoke
    assert "Tool value: custom" in out
    assert "Tool nested.field: updated" in out
    assert "Tool nested.another_field: 999" in out


def test_factory_function_params_and_invoke_args(with_extra_imports_root):
    """
    Behavior: factory function parameters are bound using type info and passed to invoke() as strings.
    SimpleTestTool prints the 'Args' dict it receives in invoke(); assert values are stringified.
    """
    result = run_tool(
        "mypackage.tools.make_test_tool",
        "run=my_test",
        "count=99",
        "value=override",
    )
    assert result.returncode == 0
    out = result.stdout + result.stderr
    # SimpleTestTool.invoke prints "Args: {...}" and its own fields
    # Runner passes only factory-function args (not overrides) to invoke(), as strings
    assert "Args: {'run': 'my_test', 'count': '99'}" in out
    assert "Tool value: override" in out  # override applied to the tool instance
