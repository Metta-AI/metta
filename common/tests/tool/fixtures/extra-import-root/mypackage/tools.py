from typing import Any

from pydantic import Field

from metta.common.tool import Tool


class TestTool(Tool):
    def invoke(self, args, overrides):
        print("TestTool invoked")
        return 0


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
