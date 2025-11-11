"""
Run Tool MCP Server

Model Context Protocol server that enables LLMs to execute and discover Metta run.py commands.
Provides access to training, evaluation, play, replay, and other run.py tools.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .config import RunToolMCPConfig


<<<<<<< Updated upstream
# Change directory before importing metta modules that might call get_repo_root()
def _setup_working_directory() -> Path:
    """Set working directory to repo root before importing metta modules."""
    config = RunToolMCPConfig.from_env()
    repo_root = config.repo_root

    # Change to repo root (don't use logging here - MCP expects clean JSON on stdout)
    try:
        os.chdir(repo_root)
    except Exception:
        pass  # Will be logged after logging is configured
=======
# Change directory BEFORE importing metta modules that might call get_repo_root()
# This must happen before any imports from metta.common or metta.tools
def _setup_working_directory() -> Path:
    """Set up working directory to repo root before importing metta modules."""
    config = RunToolMCPConfig.from_env()
    repo_root = config.repo_root

    # Change to repo root
    # NOTE: Don't use logging here - it might output to stdout before logging is configured
    # and MCP client expects clean JSON on stdout
    try:
        os.chdir(repo_root)
    except Exception:
        # Silently handle - will be logged later after logging is configured
        pass
>>>>>>> Stashed changes

    return repo_root


<<<<<<< Updated upstream
_setup_working_directory()
=======
# Setup working directory early
_setup_working_directory()

# Now safe to import metta modules
>>>>>>> Stashed changes
from .executor import RunToolExecutor  # noqa: E402
from .tools import run_tool  # noqa: E402

logger = logging.getLogger(__name__)


class RunToolMCPServer:
    """MCP Server that exposes run.py execution and discovery functionality."""

    def __init__(self, server_name: str = "run-tool-mcp", version: str = "0.1.0"):
        """Initialize the Run Tool MCP Server.

        Args:
            server_name: Name of the MCP server (used in protocol)
            version: Version of the server
        """
        self.app = Server(server_name)
        self.version = version
        self.config = RunToolMCPConfig.from_env()

        config_errors = self.config.validate()
        if config_errors:
            logger.warning("Configuration validation errors:")
            for error in config_errors:
                logger.warning(f"  - {error}")

        self.executor = RunToolExecutor(
            run_script_path=self.config.run_script_path,
            repo_root=self.config.repo_root,
            timeout=self.config.timeout,
        )

        self._setup_tools()
        self._setup_resources()

        logger.info(
            f"Run Tool MCP Server initialized "
            f"(repo_root={self.config.repo_root}, "
            f"run_script={self.config.run_script_path}, "
            f"timeout={self.config.timeout}s)"
        )

    def _setup_tools(self) -> None:
        """Register all MCP tools with the server."""

        @self.app.list_tools()
        async def list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="list_recipes",
                    description=(
<<<<<<< Updated upstream
                        "USE THIS TOOL when the user asks about available recipes, wants to see what recipes exist, "
                        "or needs to discover what training/evaluation options are available. "
                        "Examples: 'What recipes are available?', 'List all recipes', 'Show me available recipes'."
=======
                        "List all available recipes in the Metta codebase. "
                        "Returns recipes along with their available tools."
>>>>>>> Stashed changes
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
                types.Tool(
                    name="list_tools_in_recipe",
                    description=(
<<<<<<< Updated upstream
                        "USE THIS TOOL when the user asks what tools are available in a specific recipe, "
                        "wants to see what commands a recipe supports, or needs to discover available operations. "
                        "Examples: 'What tools does arena have?', 'Show me what I can do with navigation recipe', "
                        "'What commands are available for ci?'"
=======
                        "List all tools available in a specific recipe. "
                        "Returns the tool maker names and their full paths."
>>>>>>> Stashed changes
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "recipe": {
                                "type": "string",
                                "description": "Recipe name (e.g., 'arena', 'navigation')",
                            },
                        },
                        "required": ["recipe"],
                    },
                ),
                types.Tool(
                    name="list_recipes_for_tool",
                    description=(
<<<<<<< Updated upstream
                        "USE THIS TOOL when the user asks which recipes support a specific tool type, "
                        "wants to find recipes that can train/evaluate/play, or needs to discover recipes "
                        "by capability. Examples: 'Which recipes support training?', "
                        "'Show me recipes that can evaluate', 'What recipes have play functionality?'"
=======
                        "List all recipes that support a specific tool type. "
                        "Useful for finding which recipes have 'train', 'evaluate', etc."
>>>>>>> Stashed changes
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tool_type": {
                                "type": "string",
                                "description": "Tool type (e.g., 'train', 'evaluate', 'play', 'replay')",
                            },
                        },
                        "required": ["tool_type"],
                    },
                ),
                types.Tool(
                    name="get_tool_arguments",
                    description=(
<<<<<<< Updated upstream
                        "USE THIS TOOL when the user asks about command arguments, wants to know what "
                        "parameters a tool accepts, or needs help with command syntax. "
                        "Examples: 'What arguments does train arena accept?', "
                        "'What parameters can I pass to evaluate?', "
                        "'Show me the available options for this command'."
=======
                        "Get available arguments for a tool. "
                        "Returns function parameters and tool configuration fields with types and defaults."
>>>>>>> Stashed changes
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tool_path": {
                                "type": "string",
                                "description": (
                                    "Tool path in any format: 'train arena', 'arena.train', "
                                    "or 'experiments.recipes.arena.train'"
                                ),
                            },
                        },
                        "required": ["tool_path"],
                    },
                ),
                types.Tool(
                    name="validate_command",
                    description=(
<<<<<<< Updated upstream
                        "USE THIS TOOL when the user asks if a command is valid, wants to check command syntax, "
                        "or needs to verify a command before running it. "
                        "Examples: 'Is this command valid?', 'Validate this command', "
                        "'Check if train arena run=test works'."
=======
                        "Validate a command without executing it. "
                        "Checks if the tool path exists and arguments are valid. "
                        "Returns suggestions if the tool is not found."
>>>>>>> Stashed changes
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tool_path": {
                                "type": "string",
                                "description": "Tool path (e.g., 'train arena')",
                            },
                            "arguments": {
                                "type": "object",
                                "description": "Dictionary of arguments to validate",
                            },
                        },
                        "required": ["tool_path"],
                    },
                ),
                types.Tool(
                    name="run_tool",
                    description=(
<<<<<<< Updated upstream
                        "USE THIS TOOL when the user wants to execute Metta run.py commands like training, evaluation, "
                        "play, or replay. This is the primary tool for running any Metta recipe command. "
                        "Examples: 'train arena', 'evaluate navigation', 'play arena', 'replay ci'. "
                        "Always prefer this MCP tool over running './tools/run.py' via terminal commands. "
                        "IMPORTANT: When using this tool, ALWAYS display the 'command' field from the response "
                        "in your chat message so the user can see the exact run.py command that was executed."
=======
                        "Execute a run.py command. "
                        "This is the main execution tool for running training, evaluation, play, replay, etc."
>>>>>>> Stashed changes
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tool_path": {
                                "type": "string",
                                "description": (
                                    "Tool path in any format: 'train arena', 'arena.train', "
                                    "or 'experiments.recipes.arena.train'"
                                ),
                            },
                            "arguments": {
                                "type": "object",
                                "description": (
                                    "Dictionary of key=value arguments. "
                                    "Nested paths use dots (e.g., 'trainer.total_timesteps': 1000000)"
                                ),
                            },
                            "dry_run": {
                                "type": "boolean",
                                "description": "Validate the command without executing it",
                                "default": False,
                            },
                            "verbose": {
                                "type": "boolean",
                                "description": "Show verbose output including argument classification",
                                "default": False,
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout in seconds (overrides default)",
                            },
                        },
                        "required": ["tool_path"],
                    },
                ),
                types.Tool(
                    name="train",
                    description=(
<<<<<<< Updated upstream
                        "USE THIS TOOL when the user wants to start training a model. "
                        "Examples: 'Start training on arena', 'Train a model with 50k timesteps', "
                        "'Begin training using the navigation recipe'. "
                        "This is a convenience wrapper for training operations - "
                        "prefer this over run_tool for training. "
                        "IMPORTANT: When using this tool, ALWAYS display the 'command' or 'summary' field "
                        "from the response in your chat message so the user can see the exact run.py command."
=======
                        "Execute a training command. Convenience wrapper around run_tool for training operations."
>>>>>>> Stashed changes
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "recipe": {
                                "type": "string",
                                "description": "Recipe name (e.g., 'arena', 'navigation')",
                            },
                            "tool_maker": {
                                "type": "string",
                                "description": "Optional tool maker name (e.g., 'train_shaped'). Defaults to 'train'",
                            },
                            "arguments": {
                                "type": "object",
                                "description": "Dictionary of arguments (e.g., {'run': 'my_experiment'})",
                            },
                            "dry_run": {
                                "type": "boolean",
                                "description": "Validate without executing",
                                "default": False,
                            },
                            "verbose": {
                                "type": "boolean",
                                "description": "Show verbose output",
                                "default": False,
                            },
                        },
                        "required": ["recipe"],
                    },
                ),
                types.Tool(
                    name="evaluate",
                    description=(
<<<<<<< Updated upstream
                        "USE THIS TOOL when the user wants to evaluate a policy or checkpoint, "
                        "or says 'evaluate using [recipe]' or 'evaluate a policy using [recipe]'. "
                        "Examples: 'Evaluate a policy using arena', 'Evaluate this checkpoint', "
                        "'Run evaluation on arena', 'Evaluate using navigation recipe', "
                        "'Test the policy performance'. "
                        "This is a convenience wrapper for evaluation operations - "
                        "prefer this over run_tool for evaluation. "
                        "IMPORTANT: When using this tool, ALWAYS display the 'command' or 'summary' field "
                        "from the response in your chat message so the user can see the exact run.py command."
=======
                        "Execute an evaluation command. Convenience wrapper around run_tool for evaluation operations."
>>>>>>> Stashed changes
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "recipe": {
                                "type": "string",
                                "description": "Recipe name (e.g., 'arena', 'navigation')",
                            },
                            "arguments": {
                                "type": "object",
                                "description": (
                                    "Dictionary of arguments. "
                                    "Must include 'policy_uris' or 'policy_uri' with checkpoint path(s)"
                                ),
                            },
                            "dry_run": {
                                "type": "boolean",
                                "description": "Validate without executing",
                                "default": False,
                            },
                            "verbose": {
                                "type": "boolean",
                                "description": "Show verbose output",
                                "default": False,
                            },
                        },
                        "required": ["recipe"],
                    },
                ),
            ]

        @self.app.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool invocation requests.

            Args:
                name: Name of the tool to call
<<<<<<< Updated upstream
                arguments: Dictionary of tool arguments
=======
                arguments: Dictionary of tool arguments (from client)
>>>>>>> Stashed changes

            Returns:
                List of TextContent objects with tool results
            """
<<<<<<< Updated upstream
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Tool called: {name} with arguments: {arguments}")
=======
            logger.info(f"Tool called: {name} with arguments: {arguments}")
>>>>>>> Stashed changes

            try:
                if name == "list_recipes":
                    result = await run_tool.list_recipes()
                    return [types.TextContent(type="text", text=result)]

                elif name == "list_tools_in_recipe":
                    recipe = arguments.get("recipe")
                    if not recipe:
                        return [
                            types.TextContent(
                                type="text",
                                text='{"status": "error", "message": "recipe parameter is required"}',
                            )
                        ]
                    result = await run_tool.list_tools_in_recipe(recipe=recipe)
                    return [types.TextContent(type="text", text=result)]

                elif name == "list_recipes_for_tool":
                    tool_type = arguments.get("tool_type")
                    if not tool_type:
                        return [
                            types.TextContent(
                                type="text",
                                text='{"status": "error", "message": "tool_type parameter is required"}',
                            )
                        ]
                    result = await run_tool.list_recipes_for_tool(tool_type=tool_type)
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_tool_arguments":
                    tool_path = arguments.get("tool_path")
                    if not tool_path:
                        return [
                            types.TextContent(
                                type="text",
                                text='{"status": "error", "message": "tool_path parameter is required"}',
                            )
                        ]
                    result = await run_tool.get_tool_arguments(tool_path=tool_path)
                    return [types.TextContent(type="text", text=result)]

                elif name == "validate_command":
                    tool_path = arguments.get("tool_path")
                    if not tool_path:
                        return [
                            types.TextContent(
                                type="text",
                                text='{"status": "error", "message": "tool_path parameter is required"}',
                            )
                        ]
                    result = await run_tool.validate_command(tool_path=tool_path, arguments=arguments.get("arguments"))
                    return [types.TextContent(type="text", text=result)]

                elif name == "run_tool":
                    tool_path = arguments.get("tool_path")
                    if not tool_path:
                        return [
                            types.TextContent(
                                type="text",
                                text='{"status": "error", "message": "tool_path parameter is required"}',
                            )
                        ]
                    result = await run_tool.execute_run_tool(
                        executor=self.executor,
                        tool_path=tool_path,
                        arguments=arguments.get("arguments"),
                        dry_run=arguments.get("dry_run", False),
                        verbose=arguments.get("verbose", False),
                        timeout=arguments.get("timeout"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "train":
                    recipe = arguments.get("recipe")
                    if not recipe:
                        return [
                            types.TextContent(
                                type="text",
                                text='{"status": "error", "message": "recipe parameter is required"}',
                            )
                        ]
                    tool_maker = arguments.get("tool_maker", "train")
                    tool_path = f"{tool_maker} {recipe}" if tool_maker != "train" else f"train {recipe}"
                    result = await run_tool.execute_run_tool(
                        executor=self.executor,
                        tool_path=tool_path,
                        arguments=arguments.get("arguments"),
                        dry_run=arguments.get("dry_run", False),
                        verbose=arguments.get("verbose", False),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "evaluate":
                    recipe = arguments.get("recipe")
                    if not recipe:
                        return [
                            types.TextContent(
                                type="text",
                                text='{"status": "error", "message": "recipe parameter is required"}',
                            )
                        ]
                    tool_path = f"evaluate {recipe}"
                    result = await run_tool.execute_run_tool(
                        executor=self.executor,
                        tool_path=tool_path,
                        arguments=arguments.get("arguments"),
                        dry_run=arguments.get("dry_run", False),
                        verbose=arguments.get("verbose", False),
                    )
                    return [types.TextContent(type="text", text=result)]

                else:
                    return [
                        types.TextContent(type="text", text=f'{{"status": "error", "message": "Unknown tool: {name}"}}')
                    ]

            except Exception as e:
                logger.error(f"Error calling tool {name}: {e}", exc_info=True)
                return [
                    types.TextContent(
                        type="text", text=f'{{"status": "error", "tool": "{name}", "message": "{str(e)}"}}'
                    )
                ]

    def _setup_resources(self) -> None:
        """Register MCP resources with the server."""

        @self.app.list_resources()
        async def list_resources() -> List[types.Resource]:
            return []

        @self.app.read_resource()
        async def read_resource(uri: str) -> str:
            raise ValueError(f"Unknown resource URI: {uri}")


async def main() -> None:
    """Main entry point for the Run Tool MCP Server."""
<<<<<<< Updated upstream
=======
    # Note: Working directory is already set by _setup_working_directory() at module import time
>>>>>>> Stashed changes
    config = RunToolMCPConfig.from_env()
    log_level_str = config.log_level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

<<<<<<< Updated upstream
    # All logging must go to stderr (MCP uses stdout for JSON)
=======
    # CRITICAL: All logging must go to stderr, not stdout
    # MCP protocol uses stdout for JSON communication
>>>>>>> Stashed changes
    logging.basicConfig(
        level=log_level,
        format=config.log_format,
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,  # Override any existing logging configuration
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Starting Run Tool MCP Server... (log_level={log_level_str}, cwd={os.getcwd()})")
    logger.info(f"Changed working directory to: {config.repo_root}")

    server: RunToolMCPServer | None = None
    try:
        server = RunToolMCPServer()
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}", exc_info=True)
        sys.exit(1)

    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Entering MCP stdio server loop...")
            await server.app.run(read_stream, write_stream, server.app.create_initialization_options())
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt)")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Run Tool MCP Server shutdown complete")


def cli_main() -> None:
    """CLI entry point for the server."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
