"""Tool registry for managing tool types.

Public API:
- get_tool_registry() -> ToolRegistry singleton instance

The ToolRegistry class manages tool types (classes).
Tool loading and path resolution is in tool.py.
"""

import metta.common.tool

# -----------------------------------------------------------------------------
# Tool Registry
# -----------------------------------------------------------------------------


class ToolRegistry:
    """Singleton registry mapping tool type names to Tool classes.

    Access the registry dict directly via `.name_to_tool` attribute.
    """

    _instance: ToolRegistry | None = None
    name_to_tool: dict[str, type[metta.common.tool.Tool]]  # tool_type_name -> Tool class

    def __new__(cls) -> ToolRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.name_to_tool = {}
            cls._instance._register_tools()
        return cls._instance

    def _register_tools(self) -> None:
        """Lazy import and register all tools."""
        import metta.tools.analyze
        import metta.tools.eval
        import metta.tools.eval_remote
        import metta.tools.play
        import metta.tools.replay
        import metta.tools.sweep
        import metta.tools.train

        self.register(metta.tools.train.TrainTool)
        self.register(metta.tools.eval.EvaluateTool)
        self.register(metta.tools.eval_remote.EvalRemoteTool)
        self.register(metta.tools.play.PlayTool)
        self.register(metta.tools.replay.ReplayTool)
        self.register(metta.tools.analyze.AnalysisTool)
        self.register(metta.tools.sweep.SweepTool)

    def register(self, tool_class: type[metta.common.tool.Tool]) -> None:
        """Register a tool class, using its tool_type_name() as the key."""
        tool_type = tool_class.tool_type_name()
        assert tool_type, f"Tool class {tool_class.__name__} must have a non-empty tool_type_name"
        self.name_to_tool[tool_type] = tool_class


# Global singleton - access directly
tool_registry = ToolRegistry()
