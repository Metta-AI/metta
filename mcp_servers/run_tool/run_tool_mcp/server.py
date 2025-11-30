"""Run Tool MCP Server."""

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


def _setup_working_directory() -> Path:
    """Set working directory and Python path to repo root before importing metta modules."""
    config = RunToolMCPConfig.from_env()
    repo_root = config.repo_root

    try:
        os.chdir(repo_root)
    except Exception:
        pass

    common_src = repo_root / "common" / "src"
    if common_src.exists() and str(common_src) not in sys.path:
        sys.path.insert(0, str(common_src))

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    return repo_root


_setup_working_directory()

from .descriptions import MCP_TOOL_DESCRIPTIONS  # noqa: E402
from .models import (  # noqa: E402
    ErrorResponse,
    EvaluateInput,
    GetToolArgumentsInput,
    ListRecipesForToolInput,
    ListRecipesInput,
    ListToolsInRecipeInput,
    RunToolInput,
    TrainInput,
    ValidateCommandInput,
)
from .schema import get_tool_registry_schema, pydantic_to_mcp_schema  # noqa: E402
from .utils import normalize_arguments  # noqa: E402
from .executor import RunToolExecutor  # noqa: E402
from . import tools  # noqa: E402

logger = logging.getLogger(__name__)


class RunToolMCPServer:
    """MCP Server that exposes run.py execution and discovery functionality."""

    def __init__(self, server_name: str = "run-tool-mcp", version: str = "0.1.0"):
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

    def _create_error_response(self, message: str, tool: str | None = None, **kwargs: Any) -> List[types.TextContent]:
        """Create a standardized error response using ErrorResponse model."""
        error = ErrorResponse(message=message, tool=tool, **kwargs)
        return [types.TextContent(type="text", text=error.model_dump_json(indent=2))]

    def _validate_required_param(self, param_name: str, param_value: Any) -> List[types.TextContent] | None:
        """Validate that a required parameter is provided."""
        if not param_value:
            return self._create_error_response(message=f"{param_name} parameter is required")
        return None

    def _setup_tools(self) -> None:
        """Register all MCP tools with the server."""

        @self.app.list_tools()
        async def list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="list_recipes",
                    description=MCP_TOOL_DESCRIPTIONS["list_recipes"],
                    inputSchema=pydantic_to_mcp_schema(ListRecipesInput),
                ),
                types.Tool(
                    name="list_tools_in_recipe",
                    description=MCP_TOOL_DESCRIPTIONS["list_tools_in_recipe"],
                    inputSchema=pydantic_to_mcp_schema(ListToolsInRecipeInput),
                ),
                types.Tool(
                    name="list_recipes_for_tool",
                    description=MCP_TOOL_DESCRIPTIONS["list_recipes_for_tool"],
                    inputSchema=pydantic_to_mcp_schema(ListRecipesForToolInput),
                ),
                types.Tool(
                    name="get_tool_arguments",
                    description=MCP_TOOL_DESCRIPTIONS["get_tool_arguments"],
                    inputSchema=pydantic_to_mcp_schema(GetToolArgumentsInput),
                ),
                types.Tool(
                    name="validate_command",
                    description=MCP_TOOL_DESCRIPTIONS["validate_command"],
                    inputSchema=pydantic_to_mcp_schema(ValidateCommandInput),
                ),
                types.Tool(
                    name="run_tool",
                    description=MCP_TOOL_DESCRIPTIONS["run_tool"],
                    inputSchema=pydantic_to_mcp_schema(RunToolInput),
                ),
                types.Tool(
                    name="train",
                    description=MCP_TOOL_DESCRIPTIONS["train"],
                    inputSchema=(get_tool_registry_schema("train") or pydantic_to_mcp_schema(TrainInput)),
                ),
                types.Tool(
                    name="evaluate",
                    description=MCP_TOOL_DESCRIPTIONS["evaluate"],
                    inputSchema=(get_tool_registry_schema("evaluate") or pydantic_to_mcp_schema(EvaluateInput)),
                ),
            ]

        @self.app.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool invocation requests."""
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Tool called: {name} with arguments: {arguments}")

            try:
                if name == "list_recipes":
                    result = await tools.list_recipes()
                    return [types.TextContent(type="text", text=result)]

                elif name == "list_tools_in_recipe":
                    recipe = arguments.get("recipe")
                    error_response = self._validate_required_param("recipe", recipe)
                    if error_response:
                        return error_response
                    result = await tools.list_tools_in_recipe(recipe=recipe)
                    return [types.TextContent(type="text", text=result)]

                elif name == "list_recipes_for_tool":
                    tool_type = arguments.get("tool_type")
                    error_response = self._validate_required_param("tool_type", tool_type)
                    if error_response:
                        return error_response
                    result = await tools.list_recipes_for_tool(tool_type=tool_type)
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_tool_arguments":
                    tool_path = arguments.get("tool_path")
                    error_response = self._validate_required_param("tool_path", tool_path)
                    if error_response:
                        return error_response
                    result = await tools.get_tool_arguments(tool_path=tool_path)
                    return [types.TextContent(type="text", text=result)]

                elif name == "validate_command":
                    tool_path = arguments.get("tool_path")
                    error_response = self._validate_required_param("tool_path", tool_path)
                    if error_response:
                        return error_response
                    normalized_args = normalize_arguments(arguments.get("arguments"))
                    result = await tools.validate_command(tool_path=tool_path, arguments=normalized_args)
                    return [types.TextContent(type="text", text=result)]

                elif name == "run_tool":
                    tool_path = arguments.get("tool_path")
                    error_response = self._validate_required_param("tool_path", tool_path)
                    if error_response:
                        return error_response
                    normalized_args = normalize_arguments(arguments.get("arguments"))
                    result = await tools.execute_run_tool(
                        executor=self.executor,
                        tool_path=tool_path,
                        arguments=normalized_args,
                        dry_run=arguments.get("dry_run", False),
                        verbose=arguments.get("verbose", False),
                        timeout=arguments.get("timeout"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "train":
                    recipe = arguments.get("recipe")
                    error_response = self._validate_required_param("recipe", recipe)
                    if error_response:
                        return error_response
                    tool_maker = arguments.get("tool_maker", "train")
                    tool_path = f"{tool_maker} {recipe}" if tool_maker != "train" else f"train {recipe}"
                    normalized_args = normalize_arguments(arguments.get("arguments"))
                    result = await tools.execute_run_tool(
                        executor=self.executor,
                        tool_path=tool_path,
                        arguments=normalized_args,
                        dry_run=arguments.get("dry_run", False),
                        verbose=arguments.get("verbose", False),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "evaluate":
                    recipe = arguments.get("recipe")
                    error_response = self._validate_required_param("recipe", recipe)
                    if error_response:
                        return error_response
                    tool_path = f"evaluate {recipe}"
                    normalized_args = normalize_arguments(arguments.get("arguments"))
                    result = await tools.execute_run_tool(
                        executor=self.executor,
                        tool_path=tool_path,
                        arguments=normalized_args,
                        dry_run=arguments.get("dry_run", False),
                        verbose=arguments.get("verbose", False),
                    )
                    return [types.TextContent(type="text", text=result)]

                else:
                    return self._create_error_response(message=f"Unknown tool: {name}")

            except Exception as e:
                logger.error(f"Error calling tool {name}: {e}", exc_info=True)
                return self._create_error_response(message=str(e), tool=name)

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
    config = RunToolMCPConfig.from_env()
    log_level_str = config.log_level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format=config.log_format,
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
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
