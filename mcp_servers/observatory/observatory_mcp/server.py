"""
Observatory MCP Server

Model Context Protocol server that enables LLMs to interact with the Metta Observatory backend.
Provides access to training runs, policies, evaluations, scorecards, and SQL queries.
"""

import asyncio
import logging
import sys
from typing import Any, Dict, List

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from metta.app_backend.clients.scorecard_client import ScorecardClient

from .config import ObservatoryMCPConfig
from . import tools

logger = logging.getLogger(__name__)


class ObservatoryMCPServer:
    """MCP Server that exposes Observatory backend functionality."""

    def __init__(self, server_name: str = "observatory-mcp", version: str = "0.1.0"):
        """Initialize the Observatory MCP Server.

        Args:
            server_name: Name of the MCP server (used in protocol)
            version: Version of the server
        """
        self.app = Server(server_name)
        self.version = version
        self.config = ObservatoryMCPConfig.from_env()

        config_errors = self.config.validate()
        if config_errors:
            logger.warning("Configuration validation errors:")
            for error in config_errors:
                logger.warning(f"  - {error}")

        self.scorecard_client = ScorecardClient(
            backend_url=self.config.backend_url,
            machine_token=self.config.machine_token,
        )

        self._setup_tools()
        self._setup_resources()

        logger.info(
            f"Observatory MCP Server initialized "
            f"(backend={self.config.backend_url}, "
            f"authenticated={self.config.is_authenticated()})"
        )

    def _setup_tools(self) -> None:
        """Register all MCP tools with the server."""
        @self.app.list_tools()
        async def list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="get_training_runs",
                    description="Get all training runs from the backend. Returns training runs along with their metadata (name, created_at, tags, etc.).",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
                types.Tool(
                    name="get_policies",
                    description="Get all policies and training runs from the backend. Returns both training runs and standalone (run-free) policies.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
                types.Tool(
                    name="search_policies",
                    description="Search policies with filtering and pagination. Supports filtering by name, type, tags, and user ID.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "search": {
                                "type": "string",
                                "description": "Search term for policy names (case-insensitive partial match)",
                            },
                            "policy_type": {
                                "type": "string",
                                "enum": ["training_run", "policy"],
                                "description": "Filter by policy type: 'training_run' or 'policy'",
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filter by tags (policies must have at least one matching tag)",
                            },
                            "user_id": {
                                "type": "string",
                                "description": "Filter by user ID",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (1-1000)",
                                "default": 100,
                                "minimum": 1,
                                "maximum": 1000,
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Number of results to skip",
                                "default": 0,
                                "minimum": 0,
                            },
                        },
                        "required": [],
                    },
                ),
                types.Tool(
                    name="get_eval_names",
                    description="Get available evaluation names for selected training runs and policies. Returns list of eval names in format 'eval_category/env_name'.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "training_run_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of training run IDs",
                            },
                            "run_free_policy_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of run-free policy IDs",
                            },
                        },
                        "required": ["training_run_ids", "run_free_policy_ids"],
                    },
                ),
                types.Tool(
                    name="get_available_metrics",
                    description="Get available metrics for selected policies and evaluations. Returns list of metric names that can be used for scorecard generation.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "training_run_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of training run IDs",
                            },
                            "run_free_policy_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of run-free policy IDs",
                            },
                            "eval_names": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of evaluation names (format: 'eval_category/env_name')",
                            },
                        },
                        "required": ["training_run_ids", "run_free_policy_ids", "eval_names"],
                    },
                ),
                types.Tool(
                    name="generate_scorecard",
                    description="Generate scorecard (heatmap) data showing policy performance across evaluations for a specific metric. Creates a 2D grid of policy vs evaluation performance.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "training_run_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of training run IDs",
                            },
                            "run_free_policy_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of run-free policy IDs",
                            },
                            "eval_names": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of evaluation names (format: 'eval_category/env_name')",
                            },
                            "metric": {
                                "type": "string",
                                "description": "Metric to use for scorecard (e.g., 'reward', 'score', 'episode_length')",
                            },
                            "policy_selector": {
                                "type": "string",
                                "enum": ["best", "latest"],
                                "description": "Policy selection strategy for training runs: 'best' (best performing) or 'latest' (most recent)",
                                "default": "best",
                            },
                        },
                        "required": ["training_run_ids", "run_free_policy_ids", "eval_names", "metric"],
                    },
                ),
                types.Tool(
                    name="run_sql_query",
                    description="Execute SQL query against the backend database. The query is validated and executed by the backend API. Returns query results with columns and rows.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "SQL query string to execute",
                            },
                        },
                        "required": ["sql"],
                    },
                ),
                types.Tool(
                    name="generate_ai_query",
                    description="Generate SQL query from natural language description using AI. Converts a natural language description into a SQL query that can be executed against the backend database.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "Natural language description of desired query (e.g., 'Get all training runs created in the last week')",
                            },
                        },
                        "required": ["description"],
                    },
                ),
            ]

        @self.app.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool invocation requests.

            Args:
                name: Name of the tool to call
                arguments: Dictionary of tool arguments (from client)

            Returns:
                List of TextContent objects with tool results
            """
            logger.info(f"Tool called: {name} with arguments: {arguments}")

            try:
                if name == "get_training_runs":
                    result = await tools.get_training_runs(self.scorecard_client)
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_policies":
                    result = await tools.get_policies(self.scorecard_client)
                    return [types.TextContent(type="text", text=result)]

                elif name == "search_policies":
                    result = await tools.search_policies(
                        self.scorecard_client,
                        search=arguments.get("search"),
                        policy_type=arguments.get("policy_type"),
                        tags=arguments.get("tags"),
                        user_id=arguments.get("user_id"),
                        limit=arguments.get("limit", 100),
                        offset=arguments.get("offset", 0),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_eval_names":
                    training_run_ids = arguments.get("training_run_ids", [])
                    run_free_policy_ids = arguments.get("run_free_policy_ids", [])

                    if not training_run_ids and not run_free_policy_ids:
                        return [
                            types.TextContent(
                                type="text",
                                text='{"status": "error", "message": "At least one of training_run_ids or run_free_policy_ids must be provided"}'
                            )
                        ]

                    result = await tools.get_eval_names(
                        self.scorecard_client,
                        training_run_ids=training_run_ids,
                        run_free_policy_ids=run_free_policy_ids,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_available_metrics":
                    training_run_ids = arguments.get("training_run_ids", [])
                    run_free_policy_ids = arguments.get("run_free_policy_ids", [])
                    eval_names = arguments.get("eval_names", [])

                    if not training_run_ids and not run_free_policy_ids:
                        return [
                            types.TextContent(
                                type="text",
                                text='{"status": "error", "message": "At least one of training_run_ids or run_free_policy_ids must be provided"}'
                            )
                        ]

                    if not eval_names:
                        return [
                            types.TextContent(
                                type="text",
                                text='{"status": "error", "message": "eval_names is required and cannot be empty"}'
                            )
                        ]

                    result = await tools.get_available_metrics(
                        self.scorecard_client,
                        training_run_ids=training_run_ids,
                        run_free_policy_ids=run_free_policy_ids,
                        eval_names=eval_names,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "generate_scorecard":
                    training_run_ids = arguments.get("training_run_ids", [])
                    run_free_policy_ids = arguments.get("run_free_policy_ids", [])
                    eval_names = arguments.get("eval_names", [])
                    metric = arguments.get("metric")
                    policy_selector = arguments.get("policy_selector", "best")

                    if not training_run_ids and not run_free_policy_ids:
                        return [
                            types.TextContent(
                                type="text",
                                text='{"status": "error", "message": "At least one of training_run_ids or run_free_policy_ids must be provided"}'
                            )
                        ]

                    if not eval_names:
                        return [
                            types.TextContent(
                                type="text",
                                text='{"status": "error", "message": "eval_names is required and cannot be empty"}'
                            )
                        ]

                    if not metric:
                        return [
                            types.TextContent(
                                type="text",
                                text='{"status": "error", "message": "metric is required"}'
                            )
                        ]

                    result = await tools.generate_scorecard(
                        self.scorecard_client,
                        training_run_ids=training_run_ids,
                        run_free_policy_ids=run_free_policy_ids,
                        eval_names=eval_names,
                        metric=metric,
                        policy_selector=policy_selector,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "run_sql_query":
                    sql = arguments.get("sql")
                    if not sql:
                        return [
                            types.TextContent(
                                type="text",
                                text='{"status": "error", "message": "sql parameter is required"}'
                            )
                        ]

                    result = await tools.run_sql_query(self.scorecard_client, sql=sql)
                    return [types.TextContent(type="text", text=result)]

                elif name == "generate_ai_query":
                    description = arguments.get("description")
                    if not description:
                        return [
                            types.TextContent(
                                type="text",
                                text='{"status": "error", "message": "description parameter is required"}'
                            )
                        ]

                    result = await tools.generate_ai_query(self.scorecard_client, description=description)
                    return [types.TextContent(type="text", text=result)]

                else:
                    return [
                        types.TextContent(
                            type="text",
                            text=f'{{"status": "error", "message": "Unknown tool: {name}"}}'
                        )
                    ]

            except Exception as e:
                logger.error(f"Error calling tool {name}: {e}", exc_info=True)
                return [
                    types.TextContent(
                        type="text",
                        text=f'{{"status": "error", "tool": "{name}", "message": "{str(e)}"}}'
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
    """Main entry point for the Observatory MCP Server."""
    config = ObservatoryMCPConfig.from_env()
    log_level_str = config.log_level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format=config.log_format,
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Starting Observatory MCP Server... (log_level={log_level_str})")

    server: ObservatoryMCPServer | None = None
    try:
        server = ObservatoryMCPServer()
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}", exc_info=True)
        sys.exit(1)

    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Entering MCP stdio server loop...")
            await server.app.run(
                read_stream,
                write_stream,
                server.app.create_initialization_options()
            )
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt)")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if server is not None:
            try:
                await server.scorecard_client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")
        logger.info("Observatory MCP Server shutdown complete")


def cli_main() -> None:
    """CLI entry point for the server."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()

