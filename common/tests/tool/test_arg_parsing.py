import subprocess
from pathlib import Path
from typing import Any

from pydantic import Field

from metta.common.tool import Tool


# Test tools for verifying argument parsing
class SimpleTestTool(Tool):
    """A simple tool with a field for testing."""

    value: str = "default"
    nested: dict[str, Any] = Field(default_factory=lambda: {"field": "nested_default"})

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        print(f"Args: {args}")
        print(f"Overrides: {overrides}")
        print(f"Tool value: {self.value}")
        print(f"Tool nested.field: {self.nested['field']}")
        return 0


def make_test_tool(run: str = "default_run", count: int = 42) -> SimpleTestTool:
    """Function that creates a test tool."""
    return SimpleTestTool()


# Path to the test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures/extra-import-root"


class TestArgParsing:
    """Test automatic argument classification in run_tool.py"""

    def test_function_args_vs_overrides(self):
        """Test that function args and overrides are properly classified."""
        # This should classify 'run' as a function arg and 'value' as an override
        result = subprocess.run(
            ["tool", "mypackage.tools.make_test_tool", "run=test123", "value=override_value"],
            env={**subprocess.os.environ, "PYTHONPATH": str(FIXTURES_DIR)},
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        output = result.stdout

        # Check that args were passed to the function
        assert "Args: {'run': 'test123'}" in output
        # Check that overrides were applied
        assert "Tool value: override_value" in output
        # Check that overrides list is correct
        assert "Overrides: [('value', 'override_value')]" in output

    def test_nested_overrides(self):
        """Test that nested field overrides work properly."""
        result = subprocess.run(
            ["tool", "mypackage.tools.make_test_tool", "run=test", "nested.field=custom_nested"],
            env={**subprocess.os.environ, "PYTHONPATH": str(FIXTURES_DIR)},
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        output = result.stdout

        # Check that nested override was applied
        assert "Tool nested.field: custom_nested" in output
        # Check that the override is in the list
        assert "('nested.field', 'custom_nested')" in output

    def test_multiple_args_and_overrides(self):
        """Test multiple function args and overrides together."""
        result = subprocess.run(
            [
                "tool",
                "mypackage.tools.make_test_tool",
                "run=my_run",
                "count=100",  # function args
                "value=my_value",
                "system.device=cpu",
                "nested.field=test",  # overrides
            ],
            env={**subprocess.os.environ, "PYTHONPATH": str(FIXTURES_DIR)},
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        output = result.stdout

        # Check function args
        assert "'run': 'my_run'" in output
        assert "'count': 100" in output

        # Check overrides
        assert "Tool value: my_value" in output
        assert "Tool nested.field: test" in output

    def test_unknown_argument_error(self):
        """Test that unknown arguments produce helpful error messages."""
        result = subprocess.run(
            ["tool", "mypackage.tools.make_test_tool", "unknown_arg=value"],
            env={**subprocess.os.environ, "PYTHONPATH": str(FIXTURES_DIR)},
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        error_output = result.stderr + result.stdout

        # Check for error about unknown argument
        assert "unknown_arg" in error_output
        assert "Error" in error_output or "Unknown" in error_output

        # Check that available options are shown
        assert "Available" in error_output or "available" in error_output

    def test_verbose_mode_classification_display(self):
        """Test that --verbose shows argument classification."""
        result = subprocess.run(
            ["tool", "mypackage.tools.make_test_tool", "run=test", "value=override", "--verbose"],
            env={**subprocess.os.environ, "PYTHONPATH": str(FIXTURES_DIR)},
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        output = result.stdout

        # Check for classification table or similar output
        assert "run" in output and "Function Arg" in output
        assert "value" in output and "Override" in output

    def test_tool_class_constructor(self):
        """Test argument parsing with Tool class constructors."""
        # Using TestTool directly (which is a Tool subclass)
        result = subprocess.run(
            ["tool", "mypackage.tools.TestTool"],
            env={**subprocess.os.environ, "PYTHONPATH": str(FIXTURES_DIR)},
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "TestTool invoked" in result.stdout

    def test_empty_args(self):
        """Test that tools can be run with no arguments."""
        result = subprocess.run(
            ["tool", "mypackage.tools.make_test_tool"],
            env={**subprocess.os.environ, "PYTHONPATH": str(FIXTURES_DIR)},
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        output = result.stdout

        # Should use defaults
        assert "Tool value: default" in output
        assert "Tool nested.field: nested_default" in output

    def test_boolean_and_numeric_values(self):
        """Test parsing of different value types."""
        result = subprocess.run(
            [
                "tool",
                "mypackage.tools.make_test_tool",
                "count=999",  # numeric
                "wandb.enabled=false",  # boolean
                "threshold=3.14",  # float
            ],
            env={**subprocess.os.environ, "PYTHONPATH": str(FIXTURES_DIR)},
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        output = result.stdout

        # Check that numeric value was parsed
        assert "'count': 999" in output
