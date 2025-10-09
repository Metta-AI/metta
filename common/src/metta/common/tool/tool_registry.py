"""Tool registry for managing tool types.

Public API:
- get_tool_registry() -> ToolRegistry singleton instance

The ToolRegistry class manages tool types (classes).
Tool loading and path resolution is in tool.py.
"""

from __future__ import annotations

from metta.common.tool import Tool
from metta.tools.analyze import AnalysisTool
from metta.tools.eval import EvaluateTool
from metta.tools.eval_remote import EvalRemoteTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool

# -----------------------------------------------------------------------------
# Tool Registry
# -----------------------------------------------------------------------------


class ToolRegistry:
    """Singleton registry mapping tool type names to Tool classes.

    Access the registry dict directly via `.name_to_tool` attribute.
    """

    _instance: ToolRegistry | None = None
    name_to_tool: dict[str, type[Tool]]  # tool_type_name -> Tool class

    def __new__(cls) -> ToolRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.name_to_tool = {}
        return cls._instance

    def register(self, tool_class: type[Tool]) -> None:
        """Register a tool class, using its tool_type_name() as the key."""
        tool_type = tool_class.tool_type_name()
        assert tool_type, f"Tool class {tool_class.__name__} must have a non-empty tool_type_name"
        self.name_to_tool[tool_type] = tool_class


# Global singleton - access directly
tool_registry = ToolRegistry()

# Register all tools
tool_registry.register(TrainTool)
tool_registry.register(EvaluateTool)
tool_registry.register(EvalRemoteTool)
tool_registry.register(PlayTool)
tool_registry.register(ReplayTool)
tool_registry.register(AnalysisTool)
tool_registry.register(SweepTool)
