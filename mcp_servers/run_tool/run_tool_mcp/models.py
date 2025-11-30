"""Pydantic models for MCP server responses and data structures."""

from typing import Any

from pydantic import BaseModel, Field


class ToolExecutionResult(BaseModel):
    """Result of tool execution."""

    success: bool = Field(description="Whether the execution succeeded")
    exit_code: int = Field(description="Exit code from tool execution")
    stdout: str = Field(default="", description="Standard output from tool")
    stderr: str = Field(default="", description="Standard error from tool")
    command: str = Field(description="Equivalent CLI command that was executed")
    error: str | None = Field(default=None, description="Error type if execution failed")
    summary: str | None = Field(default=None, description="Human-readable summary of the execution")


class RecipeInfo(BaseModel):
    """Information about a recipe."""

    name: str = Field(description="Short name of the recipe")
    module_name: str = Field(description="Full module path of the recipe")
    tools: list[str] = Field(description="List of tool maker names in this recipe")


class RecipeListResponse(BaseModel):
    """Response containing list of recipes."""

    recipes: list[RecipeInfo] = Field(description="List of available recipes")


class ToolListResponse(BaseModel):
    """Response containing tools in a recipe."""

    recipe: str = Field(description="Recipe name")
    module_name: str = Field(description="Full module path")
    tools: list[str] = Field(description="List of tool maker names")
    tool_paths: list[str] = Field(description="Full tool paths (recipe.tool)")
    suggestions: list[str] | None = Field(default=None, description="Suggested recipes if not found")


class RecipeForToolInfo(BaseModel):
    """Information about a recipe that supports a tool type."""

    recipe: str = Field(description="Recipe name")
    module_name: str = Field(description="Full module path")
    tool_makers: list[str] = Field(description="List of tool maker names that provide this tool type")


class RecipesForToolResponse(BaseModel):
    """Response containing recipes that support a tool type."""

    tool_type: str = Field(description="Tool type name")
    recipes: list[RecipeForToolInfo] = Field(description="List of recipes supporting this tool type")


class ToolArgumentInfo(BaseModel):
    """Information about a tool argument."""

    type: str = Field(description="Type of the argument")
    default: str | None = Field(default=None, description="Default value as string")
    required: bool | None = Field(default=None, description="Whether the argument is required")
    default_factory: bool | None = Field(default=None, description="Whether default comes from a factory")


class ToolArgumentsResponse(BaseModel):
    """Response containing tool argument information."""

    tool_path: str | None = Field(default=None, description="Normalized tool path")
    original_path: str | None = Field(default=None, description="Original tool path provided")
    module: str | None = Field(default=None, description="Module where tool is defined")
    name: str | None = Field(default=None, description="Name of the tool maker")
    function_parameters: dict[str, ToolArgumentInfo] | None = Field(
        default=None, description="Function parameters if tool maker is a function"
    )
    tool_config_fields: dict[str, ToolArgumentInfo] | None = Field(
        default=None, description="Tool configuration fields if tool maker is a Tool class"
    )
    normalized_path: str | None = Field(default=None, description="Normalized path for error cases")


class ValidationResponse(BaseModel):
    """Response from command validation."""

    valid: bool = Field(description="Whether the command is valid")
    tool_path: str | None = Field(default=None, description="Normalized tool path")
    original_path: str | None = Field(default=None, description="Original tool path")
    module: str | None = Field(default=None, description="Module where tool is defined")
    name: str | None = Field(default=None, description="Name of the tool maker")
    error: str | None = Field(default=None, description="Error message if validation failed")
    warnings: list[str] | None = Field(default=None, description="Warnings about the command")
    suggestions: list[str] | None = Field(default=None, description="Suggested alternatives if tool not found")


class ErrorResponse(BaseModel):
    """Error response."""

    status: str = Field(default="error", description="Status indicator")
    message: str = Field(description="Error message")
    tool: str | None = Field(default=None, description="Tool name that caused the error")
    suggestions: list[str] | None = Field(default=None, description="Suggested alternatives")
    available_tool_types: list[str] | None = Field(default=None, description="Available tool types")
    normalized_path: str | None = Field(default=None, description="Normalized tool path")


class ListRecipesInput(BaseModel):
    """Input for list_recipes tool."""

    pass


class ListToolsInRecipeInput(BaseModel):
    """Input for list_tools_in_recipe tool."""

    recipe: str = Field(description="Recipe name (e.g., 'arena', 'navigation')")


class ListRecipesForToolInput(BaseModel):
    """Input for list_recipes_for_tool tool."""

    tool_type: str = Field(description="Tool type (e.g., 'train', 'evaluate', 'play', 'replay')")


class GetToolArgumentsInput(BaseModel):
    """Input for get_tool_arguments tool."""

    tool_path: str = Field(
        description="Tool path in any format: 'train arena', 'arena.train', or 'experiments.recipes.arena.train'"
    )


class ValidateCommandInput(BaseModel):
    """Input for validate_command tool."""

    tool_path: str = Field(description="Tool path (e.g., 'train arena')")
    arguments: dict[str, Any] | None = Field(default=None, description="Dictionary of arguments to validate")


class RunToolInput(BaseModel):
    """Input for run_tool tool."""

    tool_path: str = Field(
        description="Tool path in any format: 'train arena', 'arena.train', or 'experiments.recipes.arena.train'"
    )
    arguments: dict[str, Any] | None = Field(
        default=None,
        description="Dictionary of key=value arguments. Nested paths use dots (e.g., 'trainer.total_timesteps': 1000000)",
    )
    dry_run: bool = Field(default=False, description="Validate the command without executing it")
    verbose: bool = Field(default=False, description="Show verbose output including argument classification")
    timeout: int | None = Field(default=None, description="Timeout in seconds (overrides default)")


class TrainInput(BaseModel):
    """Input for train tool."""

    recipe: str = Field(description="Recipe name (e.g., 'arena', 'navigation')")
    tool_maker: str | None = Field(
        default=None, description="Optional tool maker name (e.g., 'train_shaped'). Defaults to 'train'"
    )
    arguments: dict[str, Any] | None = Field(default=None, description="Dictionary of arguments (e.g., {'run': 'my_experiment'})")
    dry_run: bool = Field(default=False, description="Validate without executing")
    verbose: bool = Field(default=False, description="Show verbose output")


class EvaluateInput(BaseModel):
    """Input for evaluate tool."""

    recipe: str = Field(description="Recipe name (e.g., 'arena', 'navigation')")
    arguments: dict[str, Any] | None = Field(
        default=None,
        description="Dictionary of arguments. Must include 'policy_uris' or 'policy_uri' with checkpoint path(s)",
    )
    dry_run: bool = Field(default=False, description="Validate without executing")
    verbose: bool = Field(default=False, description="Show verbose output")
