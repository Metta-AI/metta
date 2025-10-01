"""Tool registry for managing auto-inferable tools.

This module must remain import-minimal to avoid circular dependencies.
It should only import from typing and standard library.

Future improvements:
- Consider caching alias maps to avoid rebuilding on every call
- Could extract tool info (name, aliases) into a dataclass for cleaner access
- Potential to make registration automatic via __init_subclass__ instead of manual
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
        # tool_name is guaranteed to exist by Tool.__init_subclass__ validation
        tool_name = tool_class.tool_name
        assert tool_name is not None, f"Tool class {tool_class.__name__} has no tool_name (validation failed?)"

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
            # tool_name is guaranteed to exist by Tool.__init_subclass__
            aliases = tool_class.tool_aliases
            if aliases:
                mapping[tool_class.tool_name] = list(aliases)  # type: ignore[index]
        return mapping

    def get_tool_name_map(self) -> dict[str, str]:
        """Map of every supported tool name and alias -> canonical name.

        Built from Tool.tool_name and Tool.tool_aliases on all registered tools.
        """
        mapping: dict[str, str] = {}

        # Build mapping from alias map to avoid duplication
        alias_map = self.get_tool_aliases()

        # Add canonical names pointing to themselves
        for tool_name in self._registry.keys():
            mapping[tool_name] = tool_name

        # Add aliases pointing to canonical names
        for canonical, aliases in alias_map.items():
            for alias in aliases:
                mapping[alias] = canonical

        return mapping

    def clear(self) -> None:
        """Clear the registry. Mainly for testing."""
        self._registry.clear()


# Global singleton instance for convenient access
_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global ToolRegistry singleton instance."""
    return _registry
