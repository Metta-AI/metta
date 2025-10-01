from pydantic import Field

from metta.common.tool import Tool
from mettagrid.base_config import Config


class NestedConfig(Config):
    """A nested configuration object for testing."""

    field: str = "nested_default"
    another_field: int = 100


class TestTool(Tool):
    tool_name = "test"

    def invoke(self, args):
        print("TestTool invoked")
        return 0


class SimpleTestTool(Tool):
    """A simple tool with a field for testing."""

    tool_name = "simple_test"
    value: str = "default"
    nested: NestedConfig = Field(default_factory=NestedConfig)

    def invoke(self, args: dict[str, str]) -> int | None:
        print(f"Args: {args}")
        print(f"Tool value: {self.value}")
        print(f"Tool nested.field: {self.nested.field}")
        print(f"Tool nested.another_field: {self.nested.another_field}")
        return 0


def make_test_tool(run: str = "default_run", count: int = 42) -> SimpleTestTool:
    """Function that creates a test tool."""
    return SimpleTestTool()


class RequiredFieldTool(Tool):
    """Tool with a required field (no default). Used to verify constructor validation."""

    tool_name = "required_field"
    x: int

    def invoke(self, args: dict[str, str]) -> int | None:
        # Print the value so tests can assert behavior via subprocess output
        print(self.x)
        return 0
