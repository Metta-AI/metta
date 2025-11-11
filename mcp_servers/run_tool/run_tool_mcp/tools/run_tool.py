"""Tool functions for run.py execution and discovery."""

import json
import logging
from typing import Any

from metta.common.tool.recipe_registry import recipe_registry
from metta.common.tool.tool_path import parse_two_token_syntax, resolve_and_load_tool_maker
from metta.common.tool.tool_registry import tool_registry

logger = logging.getLogger(__name__)


def _normalize_tool_path(tool_path: str) -> str:
    """Normalize tool path to handle two-token syntax like 'train arena'.

    This function handles the same syntax as the CLI run.py:
    - 'train arena' -> 'arena.train'
    - 'arena.train' -> 'arena.train' (no change)
    - 'experiments.recipes.arena.train' -> 'experiments.recipes.arena.train' (no change)

    Args:
        tool_path: Tool path in any format

    Returns:
        Normalized tool path (e.g., 'arena.train')
    """
    # Split by space to check for two-token syntax
    parts = tool_path.strip().split(None, 1)

    if len(parts) == 2:
        first_token, second_token = parts
        # Check if second token looks like an argument (contains = or starts with -)
        if "=" not in second_token and not second_token.startswith("-"):
            # This looks like two-token syntax (e.g., "train arena")
            resolved_path, _ = parse_two_token_syntax(first_token, second_token)
            return resolved_path

    # Already in normalized format or single token
    return tool_path


async def list_recipes() -> str:
    """List all available recipes.

    Returns:
        JSON string with list of recipes and their tools
    """
    try:
        recipes = recipe_registry.get_all()
        result = {
            "recipes": [
                {
                    "name": recipe.short_name,
                    "module_name": recipe.module_name,
                    "tools": sorted(recipe.get_all_tool_maker_names()),
                }
                for recipe in sorted(recipes, key=lambda r: r.module_name)
            ]
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error listing recipes: {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e)})


async def list_tools_in_recipe(recipe: str) -> str:
    """List all tools available in a specific recipe.

    Args:
        recipe: Recipe name (e.g., 'arena')

    Returns:
        JSON string with list of tools
    """
    try:
        recipe_obj = recipe_registry.get(recipe)
        if not recipe_obj:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Recipe '{recipe}' not found",
                    "suggestions": [
                        r.short_name for r in recipe_registry.get_all() if recipe.lower() in r.short_name.lower()
                    ][:5],
                }
            )

        tools = sorted(recipe_obj.get_all_tool_maker_names())
        result = {
            "recipe": recipe_obj.short_name,
            "module_name": recipe_obj.module_name,
            "tools": tools,
            "tool_paths": [f"{recipe_obj.short_name}.{tool}" for tool in tools],
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error listing tools in recipe {recipe}: {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e)})


async def list_recipes_for_tool(tool_type: str) -> str:
    """List all recipes that support a specific tool type.

    Args:
        tool_type: Tool type (e.g., 'train', 'evaluate')

    Returns:
        JSON string with list of recipes supporting the tool
    """
    try:
        recipes = recipe_registry.get_all()
        matching_recipes = []

        for recipe in sorted(recipes, key=lambda r: r.module_name):
            makers = recipe.get_makers_for_tool(tool_type)
            if makers:
                matching_recipes.append(
                    {
                        "recipe": recipe.short_name,
                        "module_name": recipe.module_name,
                        "tool_makers": [name for name, _ in makers],
                    }
                )

        if not matching_recipes:
            # Check if tool type exists
            if tool_type not in tool_registry.name_to_tool:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Tool type '{tool_type}' not found",
                        "available_tool_types": list(tool_registry.name_to_tool.keys()),
                    }
                )

        result = {
            "tool_type": tool_type,
            "recipes": matching_recipes,
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error listing recipes for tool {tool_type}: {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e)})


async def get_tool_arguments(tool_path: str) -> str:
    """Get available arguments for a tool using --help.

    Args:
        tool_path: Tool path (e.g., 'train arena', 'arena.train')

    Returns:
        JSON string with tool arguments information
    """
    try:
        # Normalize tool path to handle two-token syntax
        normalized_path = _normalize_tool_path(tool_path)

        # Try to resolve the tool maker to get type information
        tool_maker = resolve_and_load_tool_maker(normalized_path)
        if not tool_maker:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Tool '{tool_path}' not found",
                    "normalized_path": normalized_path,
                }
            )

        import inspect

        from metta.common.tool import Tool

        result: dict[str, Any] = {
            "tool_path": normalized_path,
            "original_path": tool_path,
            "module": tool_maker.__module__,
            "name": tool_maker.__name__,
        }

        # Get function parameters if it's a function
        if inspect.isfunction(tool_maker) or inspect.ismethod(tool_maker):
            sig = inspect.signature(tool_maker)
            result["function_parameters"] = {}
            for name, param in sig.parameters.items():
                param_info: dict[str, Any] = {
                    "type": str(param.annotation) if param.annotation != inspect._empty else "Any",
                }
                if param.default != inspect._empty:
                    param_info["default"] = str(param.default)
                else:
                    param_info["required"] = True
                result["function_parameters"][name] = param_info

        # Get tool fields if it returns a Tool
        if inspect.isclass(tool_maker) and issubclass(tool_maker, Tool):
            # It's a Tool class
            tool_class = tool_maker
        else:
            # Try to call it to get the tool instance
            try:
                sig = inspect.signature(tool_maker)
                kwargs = {}
                for name, param in sig.parameters.items():
                    if param.default != inspect._empty:
                        kwargs[name] = param.default
                tool_instance = tool_maker(**kwargs)
                tool_class = type(tool_instance)
            except Exception:
                tool_class = None

        if tool_class and issubclass(tool_class, Tool):
            # Get Pydantic model fields
            tool_fields: dict[str, Any] = {}
            for field_name, field in tool_class.model_fields.items():
                field_info: dict[str, Any] = {
                    "type": str(field.annotation),
                }
                if field.default is not None:
                    field_info["default"] = str(field.default)
                elif field.default_factory:
                    field_info["default_factory"] = True
                else:
                    field_info["required"] = True
                tool_fields[field_name] = field_info

            result["tool_config_fields"] = tool_fields

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error getting tool arguments for {tool_path}: {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e)})


async def validate_command(tool_path: str, arguments: dict[str, Any] | None = None) -> str:
    """Validate a command without executing it.

    Args:
        tool_path: Tool path (e.g., 'train arena', 'arena.train')
        arguments: Dictionary of arguments

    Returns:
        JSON string with validation result
    """
    try:
        # Normalize tool path to handle two-token syntax
        normalized_path = _normalize_tool_path(tool_path)

        tool_maker = resolve_and_load_tool_maker(normalized_path)
        if not tool_maker:
            # Try to find similar recipes
            recipes = recipe_registry.get_all()
            suggestions = [
                r.short_name
                for r in recipes
                if tool_path.lower() in r.short_name.lower() or r.short_name.lower() in tool_path.lower()
            ][:5]

            return json.dumps(
                {
                    "valid": False,
                    "error": f"Tool '{tool_path}' not found",
                    "normalized_path": normalized_path,
                    "suggestions": suggestions,
                }
            )

        # Basic validation - check if tool maker can be loaded
        result = {
            "valid": True,
            "tool_path": normalized_path,
            "original_path": tool_path,
            "module": tool_maker.__module__,
            "name": tool_maker.__name__,
        }

        # If arguments provided, try to validate them
        if arguments:
            import inspect

            if inspect.isfunction(tool_maker) or inspect.ismethod(tool_maker):
                sig = inspect.signature(tool_maker)
                param_names = set(sig.parameters.keys())
                provided_keys = set(arguments.keys())

                # Check for unknown function parameters
                unknown_params = provided_keys - param_names
                if unknown_params:
                    result["warnings"] = result.get("warnings", [])
                    result["warnings"].append(f"Unknown function parameters: {unknown_params}")

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error validating command {tool_path}: {e}", exc_info=True)
        return json.dumps({"valid": False, "error": str(e)})


async def execute_run_tool(
    executor: Any,
    tool_path: str,
    arguments: dict[str, Any] | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    timeout: int | None = None,
) -> str:
    """Execute a run.py command.

    Args:
        executor: RunToolExecutor instance
        tool_path: Tool path (e.g., 'train arena')
        arguments: Dictionary of arguments
        dry_run: If True, validate without executing
        verbose: If True, show verbose output
        timeout: Override default timeout

    Returns:
        JSON string with execution result including command summary
    """
    # Build preview command for display
    resolved_path, _ = parse_two_token_syntax(tool_path, None)
    preview_args = []
    if arguments:
        for key, value in arguments.items():
            if value is not None:
                if isinstance(value, bool):
                    value_str = str(value).lower()
                elif isinstance(value, (list, dict)):
                    value_str = json.dumps(value)
                else:
                    value_str = str(value)
                preview_args.append(f"{key}={value_str}")

    preview_command = "./tools/run.py " + resolved_path
    if preview_args:
        preview_command += " " + " ".join(preview_args)

    result = await executor.execute(
        tool_path=tool_path,
        arguments=arguments,
        dry_run=dry_run,
        verbose=verbose,
        timeout=timeout,
    )

    # Add summary fields for command visibility
    command = result.get("command", preview_command)
    result["command"] = command

    if dry_run:
        result["summary"] = f"Would execute: {command} (dry run - validation only)"
        result["preview"] = command
    else:
        result["summary"] = f"Executed: {command}"
        result["command_executed"] = command

    return json.dumps(result, indent=2)
