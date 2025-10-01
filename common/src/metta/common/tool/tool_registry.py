"""Tool registry for managing auto-inferable tools.

This module must remain import-minimal to avoid circular dependencies.
It should only import from typing and standard library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metta.common.tool import Tool


class ToolRegistry:
    """Singleton registry for tools that support auto-inference from recipes.

    Tools register themselves by setting `supports_auto_factory = True` as a ClassVar.
    Registration happens automatically via Tool.__init_subclass__.

    This is a singleton - all methods operate on a shared instance.
    """

    _instance: ToolRegistry | None = None
    _registry: dict[str, type[Tool]]

    def __new__(cls) -> ToolRegistry:
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
        return cls._instance

    def register(self, tool_class: type[Tool]) -> None:
        """Register a tool class that supports auto-factory inference.

        Args:
            tool_class: The tool class to register (must have tool_name set)
        """
        tool_name = getattr(tool_class, "tool_name", None)
        if not tool_name:
            raise ValueError(f"Tool class {tool_class.__name__} must define tool_name to be registered")

        if tool_name in self._registry:
            # Allow re-registration of the same class (e.g., during testing or reloading)
            existing = self._registry[tool_name]
            if existing is not tool_class:
                raise ValueError(
                    f"Tool name '{tool_name}' already registered by {existing.__name__}, "
                    f"cannot register {tool_class.__name__}"
                )
        self._registry[tool_name] = tool_class

    def get_all_tools(self) -> dict[str, type[Tool]]:
        """Get all registered tool classes mapped by their canonical name."""
        return dict(self._registry)

    def get_factory_tools(self) -> tuple[type[Tool], ...]:
        """Get tuple of all tools that support auto-factory inference.

        Returns tools in registration order.
        """
        return tuple(self._registry.values())

    def get_tool_aliases(self) -> dict[str, list[str]]:
        """Build alias map (canonical -> aliases) from registered tools.

        Only includes tools that declare aliases via Tool.tool_aliases.
        """
        mapping: dict[str, list[str]] = {}
        for tool_class in self._registry.values():
            name = getattr(tool_class, "tool_name", None)
            aliases = getattr(tool_class, "tool_aliases", [])
            if name and aliases:
                mapping[name] = list(aliases)
        return mapping

    def get_tool_name_map(self) -> dict[str, str]:
        """Map of every supported tool name and alias -> canonical name.

        Built from Tool.tool_name and Tool.tool_aliases on all registered tools.
        """
        mapping: dict[str, str] = {}
        for tool_class in self._registry.values():
            tool_name = getattr(tool_class, "tool_name", None)
            if not tool_name:
                continue
            mapping[tool_name] = tool_name
            for alias in getattr(tool_class, "tool_aliases", []) or []:
                mapping[alias] = tool_name
        return mapping

    def clear(self) -> None:
        """Clear the registry. Mainly for testing."""
        self._registry.clear()


# Global singleton instance for convenient access
_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global ToolRegistry singleton instance."""
    return _registry
