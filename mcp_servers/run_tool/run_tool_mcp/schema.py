"""Schema generation utilities for MCP server."""

from typing import Any

from metta.common.tool.tool_registry import tool_registry


def pydantic_to_mcp_schema(model_class: type) -> dict[str, Any]:
    """Convert a Pydantic model to MCP inputSchema format."""
    schema = model_class.model_json_schema()

    mcp_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": schema.get("required", []),
    }

    if "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            mcp_prop: dict[str, Any] = {}
            if "type" in prop_schema:
                mcp_prop["type"] = prop_schema["type"]
            if "description" in prop_schema:
                mcp_prop["description"] = prop_schema["description"]
            if "default" in prop_schema:
                mcp_prop["default"] = prop_schema["default"]
            mcp_schema["properties"][prop_name] = mcp_prop

    return mcp_schema


def get_tool_registry_schema(tool_type: str) -> dict[str, Any] | None:
    """Generate MCP schema from ToolRegistry class for train/evaluate wrappers."""
    tool_class = tool_registry.name_to_tool.get(tool_type)
    if not tool_class:
        return None

    tool_schema = tool_class.model_json_schema()
    tool_description = None
    if hasattr(tool_class, "model_fields") and "description" in tool_class.model_fields:
        field = tool_class.model_fields["description"]
        if field.description:
            tool_description = field.description

    wrapper_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "recipe": {
                "type": "string",
                "description": "Recipe name (e.g., 'arena', 'navigation')",
            },
            "arguments": {
                "type": "object",
                "description": tool_description or f"Dictionary of arguments matching {tool_class.__name__} fields",
                "properties": {},
            },
            "dry_run": {
                "type": "boolean",
                "default": False,
                "description": "Validate without executing",
            },
            "verbose": {
                "type": "boolean",
                "default": False,
                "description": "Show verbose output",
            },
        },
        "required": ["recipe"],
    }

    if tool_type == "train":
        wrapper_schema["properties"]["tool_maker"] = {
            "type": "string",
            "description": "Optional tool maker name (e.g., 'train_shaped'). Defaults to 'train'",
        }

    if "properties" in tool_schema:
        for prop_name, prop_schema in tool_schema["properties"].items():
            mcp_prop: dict[str, Any] = {}
            if "type" in prop_schema:
                mcp_prop["type"] = prop_schema["type"]
            if "description" in prop_schema:
                mcp_prop["description"] = prop_schema["description"]
            if "default" in prop_schema:
                mcp_prop["default"] = prop_schema["default"]
            wrapper_schema["properties"]["arguments"]["properties"][prop_name] = mcp_prop

    return wrapper_schema

