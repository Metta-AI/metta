"""Tool registry for managing tool types.

Public API:
- get_tool_registry() -> ToolRegistry singleton instance

The ToolRegistry class manages tool types (classes).
Tool loading and path resolution is in tool.py.

Performance note: Tools are lazy-loaded on first access to avoid importing
heavy dependencies (torch, transformers, etc.) at CLI startup time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metta.common.tool import Tool

# Mapping from tool type name to (module_path, class_name)
# This allows lazy importing of tool classes only when needed
_TOOL_REGISTRY: dict[str, tuple[str, str]] = {
    "train": ("metta.tools.train", "TrainTool"),
    "evaluate": ("metta.tools.eval", "EvaluateTool"),
    "evaluate_remote": ("metta.tools.request_remote_eval", "RequestRemoteEvalTool"),
    "play": ("metta.tools.play", "PlayTool"),
    "replay": ("metta.tools.replay", "ReplayTool"),
    "sweep": ("metta.tools.sweep", "SweepTool"),
}

# -----------------------------------------------------------------------------
# Tool Registry
# -----------------------------------------------------------------------------


class ToolRegistry:
    """Singleton registry mapping tool type names to Tool classes.

    Tools are lazy-loaded on first access to avoid importing heavy
    dependencies (torch, transformers, etc.) at CLI startup time.

    Access via `.name_to_tool` for the dict of loaded tools, or use
    `get(name)` to lazy-load a specific tool by name.
    """

    _instance: ToolRegistry | None = None
    _loaded_tools: dict[str, type[Tool]]  # tool_type_name -> Tool class (loaded)

    def __new__(cls) -> ToolRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded_tools = {}
        return cls._instance

    @property
    def name_to_tool(self) -> dict[str, type[Tool]]:
        """Returns dict of all registered tool names to classes.

        Note: This triggers loading ALL tools. Prefer `get(name)` for
        lazy access to a single tool.
        """
        self._ensure_all_loaded()
        return self._loaded_tools

    def _ensure_all_loaded(self) -> None:
        """Load all tools if not already loaded."""
        for tool_name in _TOOL_REGISTRY:
            if tool_name not in self._loaded_tools:
                self._load_tool(tool_name)

    def _load_tool(self, tool_name: str) -> type[Tool] | None:
        """Lazy-load a single tool by name."""
        if tool_name in self._loaded_tools:
            return self._loaded_tools[tool_name]

        if tool_name not in _TOOL_REGISTRY:
            return None

        module_path, class_name = _TOOL_REGISTRY[tool_name]
        import importlib

        module = importlib.import_module(module_path)
        tool_class = getattr(module, class_name)
        self._loaded_tools[tool_name] = tool_class
        return tool_class

    def get(self, tool_name: str) -> type[Tool] | None:
        """Get a tool class by name, lazy-loading if needed."""
        return self._load_tool(tool_name)

    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool name is registered (without loading)."""
        return tool_name in _TOOL_REGISTRY

    def keys(self) -> list[str]:
        """Return all registered tool names (without loading)."""
        return list(_TOOL_REGISTRY.keys())

    def register(self, tool_class: type[Tool]) -> None:
        """Register a tool class, using its tool_type_name() as the key."""
        tool_type = tool_class.tool_type_name()
        assert tool_type, f"Tool class {tool_class.__name__} must have a non-empty tool_type_name"
        self._loaded_tools[tool_type] = tool_class


# Global singleton - access directly
tool_registry = ToolRegistry()
