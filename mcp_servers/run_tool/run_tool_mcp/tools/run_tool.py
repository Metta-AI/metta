"""Tool functions for run.py execution and discovery."""

import inspect
import json
import logging
from typing import Any

from metta.common.tool import Tool
from metta.common.tool.recipe_registry import recipe_registry
from metta.common.tool.tool_path import parse_two_token_syntax, resolve_and_load_tool_maker
from metta.common.tool.tool_registry import tool_registry

from ..models import (
    ErrorResponse,
    RecipeForToolInfo,
    RecipeInfo,
    RecipeListResponse,
    RecipesForToolResponse,
    ToolArgumentInfo,
    ToolArgumentsResponse,
    ToolListResponse,
    ValidationResponse,
)

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
    parts = tool_path.strip().split(None, 1)

    if len(parts) == 2:
        first_token, second_token = parts
        if "=" not in second_token and not second_token.startswith("-"):
            resolved_path, _ = parse_two_token_syntax(first_token, second_token)
            return resolved_path

    return tool_path


async def list_recipes() -> str:
    """List all available recipes.

    Returns:
        JSON string with list of recipes and their tools
    """
    try:
        recipes = recipe_registry.get_all()
        result = RecipeListResponse(
            recipes=[
                RecipeInfo(
                    name=recipe.short_name,
                    module_name=recipe.module_name,
                    tools=sorted(recipe.get_all_tool_maker_names()),
                )
                for recipe in sorted(recipes, key=lambda r: r.module_name)
            ]
        )
        return result.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error listing recipes: {e}", exc_info=True)
        error = ErrorResponse(message=str(e))
        return error.model_dump_json(indent=2)


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
            suggestions = [
                r.short_name for r in recipe_registry.get_all() if recipe.lower() in r.short_name.lower()
            ][:5]
            error = ErrorResponse(message=f"Recipe '{recipe}' not found", suggestions=suggestions)
            return error.model_dump_json(indent=2)

        tools = sorted(recipe_obj.get_all_tool_maker_names())
        result = ToolListResponse(
            recipe=recipe_obj.short_name,
            module_name=recipe_obj.module_name,
            tools=tools,
            tool_paths=[f"{recipe_obj.short_name}.{tool}" for tool in tools],
        )
        return result.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error listing tools in recipe {recipe}: {e}", exc_info=True)
        error = ErrorResponse(message=str(e))
        return error.model_dump_json(indent=2)


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
                    RecipeForToolInfo(
                        recipe=recipe.short_name,
                        module_name=recipe.module_name,
                        tool_makers=[name for name, _ in makers],
                    )
                )

        if not matching_recipes:
            # Check if tool type exists
            if tool_type not in tool_registry.name_to_tool:
                error = ErrorResponse(
                    message=f"Tool type '{tool_type}' not found",
                    available_tool_types=list(tool_registry.name_to_tool.keys()),
                )
                return error.model_dump_json(indent=2)

        result = RecipesForToolResponse(tool_type=tool_type, recipes=matching_recipes)
        return result.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error listing recipes for tool {tool_type}: {e}", exc_info=True)
        error = ErrorResponse(message=str(e))
        return error.model_dump_json(indent=2)


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
            error = ErrorResponse(message=f"Tool '{tool_path}' not found", normalized_path=normalized_path)
            return error.model_dump_json(indent=2)

        function_parameters: dict[str, ToolArgumentInfo] | None = None
        tool_config_fields: dict[str, ToolArgumentInfo] | None = None

        # Get function parameters if it's a function
        if inspect.isfunction(tool_maker) or inspect.ismethod(tool_maker):
            sig = inspect.signature(tool_maker)
            function_parameters = {}
            for name, param in sig.parameters.items():
                param_info = ToolArgumentInfo(
                    type=str(param.annotation) if param.annotation != inspect._empty else "Any",
                    default=str(param.default) if param.default != inspect._empty else None,
                    required=None if param.default != inspect._empty else True,
                )
                function_parameters[name] = param_info

        # Get tool fields if it returns a Tool
        if inspect.isclass(tool_maker) and issubclass(tool_maker, Tool):
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
            tool_config_fields = {}
            for field_name, field in tool_class.model_fields.items():
                field_info = ToolArgumentInfo(
                    type=str(field.annotation),
                    default=str(field.default) if field.default is not None else None,
                    required=None if (field.default is not None or field.default_factory) else True,
                    default_factory=True if field.default_factory else None,
                )
                tool_config_fields[field_name] = field_info

        result = ToolArgumentsResponse(
            tool_path=normalized_path,
            original_path=tool_path,
            module=tool_maker.__module__,
            name=tool_maker.__name__,
            function_parameters=function_parameters,
            tool_config_fields=tool_config_fields,
        )

        return result.model_dump_json(indent=2)

    except Exception as e:
        logger.error(f"Error getting tool arguments for {tool_path}: {e}", exc_info=True)
        error = ErrorResponse(message=str(e))
        return error.model_dump_json(indent=2)


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

            result = ValidationResponse(
                valid=False,
                error=f"Tool '{tool_path}' not found",
                normalized_path=normalized_path,
                suggestions=suggestions,
            )
            return result.model_dump_json(indent=2)

        warnings: list[str] | None = None

        if arguments:
            if inspect.isfunction(tool_maker) or inspect.ismethod(tool_maker):
                sig = inspect.signature(tool_maker)
                param_names = set(sig.parameters.keys())
                provided_keys = set(arguments.keys())

                # Check for unknown function parameters
                unknown_params = provided_keys - param_names
                if unknown_params:
                    warnings = [f"Unknown function parameters: {unknown_params}"]

        result = ValidationResponse(
            valid=True,
            tool_path=normalized_path,
            original_path=tool_path,
            module=tool_maker.__module__,
            name=tool_maker.__name__,
            warnings=warnings,
        )

        return result.model_dump_json(indent=2)

    except Exception as e:
        logger.error(f"Error validating command {tool_path}: {e}", exc_info=True)
        result = ValidationResponse(valid=False, error=str(e))
        return result.model_dump_json(indent=2)


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
    resolved_path, _ = parse_two_token_syntax(tool_path, None)
    preview_args = []
    if arguments:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse arguments as JSON: {arguments}")
                arguments = None
        if isinstance(arguments, dict):
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

    result_dict = await executor.execute(
        tool_path=tool_path,
        arguments=arguments,
        dry_run=dry_run,
        verbose=verbose,
        timeout=timeout,
    )

    if "command" not in result_dict or not result_dict["command"]:
        result_dict["command"] = preview_command

    return json.dumps(result_dict, indent=2)
