"""Tool registry for managing tool types.

Public API:
- get_tool_registry() -> ToolRegistry singleton instance

The ToolRegistry class manages registered tool types (classes).
Tool loading and path resolution is in tool.py.
"""

from __future__ import annotations

from metta.common.tool import Tool
from metta.tools.analyze import AnalysisTool
from metta.tools.eval import EvalTool
from metta.tools.eval_remote import EvalRemoteTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool

# -----------------------------------------------------------------------------
# Tool Registry
# -----------------------------------------------------------------------------


class ToolRegistry:
    """Registry of all available tool types."""

    _instance: ToolRegistry | None = None
    _registry: dict[str, type[Tool]]

    def __new__(cls) -> ToolRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
        return cls._instance

    def register(self, tool_class: type[Tool]) -> None:
        tool_name = tool_class.tool_name
        assert tool_name is not None, f"Tool class {tool_class.__name__} must define tool_name"
        self._registry[tool_name] = tool_class

    def get_all_tools(self) -> dict[str, type[Tool]]:
        """Get all registered tool classes mapped by their tool name."""
        return dict(self._registry)

    def clear(self) -> None:
        self._registry.clear()


# Global singleton instance
_registry = ToolRegistry()

# Register all tools
# TODO: Should we do a pkgutil scan here instead?
_registry.register(TrainTool)
_registry.register(EvalTool)
_registry.register(EvalRemoteTool)
_registry.register(PlayTool)
_registry.register(ReplayTool)
_registry.register(AnalysisTool)
_registry.register(SweepTool)


def get_tool_registry() -> ToolRegistry:
    """Get the global ToolRegistry singleton instance."""
    return _registry
