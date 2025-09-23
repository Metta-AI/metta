import os
import re

import pytest

from metta.tests_support import run_tool_in_process


def find_with_whitespace(target: str, text: str) -> bool:
    """
    Find a target string in text, allowing for arbitrary whitespace between words.
    """
    # Split the search string into words and escape regex special characters
    words = [re.escape(word) for word in target.split()]
    # Join with \s+ to match any whitespace between words
    pattern = r"\s+".join(words)
    return bool(re.search(pattern, text))


@pytest.fixture
def with_extra_imports_root(monkeypatch):
    extra_imports_root = os.path.join(os.path.dirname(__file__), "fixtures/extra-import-root")
    monkeypatch.setenv("PYTHONPATH", extra_imports_root)
    monkeypatch.syspath_prepend(extra_imports_root)


@pytest.fixture
def invoke_run_tool(monkeypatch, capsys, with_extra_imports_root):  # noqa: ANN001 - pytest fixture
    def _invoke(*args: str):
        return run_tool_in_process(*args, monkeypatch=monkeypatch, capsys=capsys, argv0="run_tool.py")

    return _invoke


def test_basic(invoke_run_tool):
    # Behavior: the tool runs and prints from invoke()
    result = invoke_run_tool("mypackage.tools.TestTool")
    assert result.returncode == 0
    combined_output = result.stdout + result.stderr
    assert find_with_whitespace("TestTool invoked", combined_output)


def test_unknown_tool(invoke_run_tool):
    # Behavior: unknown symbol yields non-zero with helpful error content
    result = invoke_run_tool("mypackage.tools.NoSuchTool")
    assert result.returncode > 0
    combined_output = result.stderr + result.stdout
    assert find_with_whitespace("has no", combined_output)
    assert "attribute" in combined_output
    assert "NoSuchTool" in combined_output


def test_unknown_module(invoke_run_tool):
    # Behavior: unknown module yields non-zero with ImportError details
    result = invoke_run_tool("mypackage.no_such_tools.TestTool")
    assert result.returncode > 0
    combined_output = result.stderr + result.stdout
    assert find_with_whitespace("No module named", combined_output)
    assert "mypackage.no_such_tools" in combined_output


def test_required_field_supported_for_tool_class(invoke_run_tool):
    """
    Behavior: required Pydantic field on a Tool subclass is validated at construction.
    Edge case from review: `x=123` must be accepted and printed.
    """
    result = invoke_run_tool("mypackage.tools.RequiredFieldTool", "x=123")
    assert result.returncode == 0
    combined_output = result.stdout + result.stderr
    assert "123" in combined_output


def test_dotted_overrides_are_nested_and_validated(invoke_run_tool):
    """
    Behavior: dotted keys are interpreted as nested config and validated at construction time.
    SimpleTestTool prints nested fields; assert they reflect our CLI.
    """
    result = invoke_run_tool(
        "mypackage.tools.SimpleTestTool",
        "value=custom",
        "nested.field=updated",
        "nested.another_field=999",
    )
    assert result.returncode == 0
    out = result.stdout + result.stderr
    # Printed by SimpleTestTool.invoke
    assert find_with_whitespace("Tool value: custom", out)
    assert find_with_whitespace("Tool nested.field: updated", out)
    assert find_with_whitespace("Tool nested.another_field: 999", out)


def test_factory_function_params_and_invoke_args(invoke_run_tool):
    """
    Behavior: factory function parameters are bound using type info and passed to invoke() as strings.
    SimpleTestTool prints the 'Args' dict it receives in invoke(); assert values are stringified.
    """
    result = invoke_run_tool(
        "mypackage.tools.make_test_tool",
        "run=my_test",
        "count=99",
        "value=override",
    )
    assert result.returncode == 0
    out = result.stdout + result.stderr
    # SimpleTestTool.invoke prints "Args: {...}" and its own fields
    # Runner passes only factory-function args (not overrides) to invoke(), as strings
    assert find_with_whitespace("Args: {'run': 'my_test', 'count': '99'}", out)
    assert find_with_whitespace("Tool value: override", out)


def test_two_token_sugar_loads_function(with_extra_imports_root):
    # Behavior: two-token form 'x y' resolves to 'y.x'
    # Use dry-run to avoid invoking the tool
    result = run_tool("make_test_tool", "mypackage.tools", "--dry-run")
    assert result.returncode == 0


def test_infer_eval_with_simulations_alias(with_extra_imports_root):
    # Behavior: eval/sim alias maps to inferred EvalTool using simulations() if present
    result = run_tool("mypackage.recipes.demo.sim", "--dry-run")
    assert result.returncode == 0


def test_infer_remote_eval_alias(with_extra_imports_root):
    # Behavior: remote aliases (eval_remote/sim_remote/evaluate_remote) are accepted for inference
    result = run_tool("mypackage.recipes.demo.eval_remote", "--dry-run")
    assert result.returncode == 0


def test_infer_eval_with_mettagrid_only(with_extra_imports_root):
    # Behavior: if only mettagrid() is present, inference still constructs an evaluation tool
    result = run_tool("mypackage.recipes.onlymg.evaluate", "--dry-run")
    assert result.returncode == 0
