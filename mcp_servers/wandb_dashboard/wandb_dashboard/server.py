"""
WandB Dashboard MCP Server

This module implements a Model Context Protocol (MCP) server that enables
Large Language Models to interact with and configure Weights & Biases dashboards.
"""

import asyncio
import logging
import sys
from typing import Any, Dict, List

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .config import WandBMCPConfig
from .tools import WandBDashboardTools, WandBDashboardToolsStub

logger = logging.getLogger(__name__)


class WandBDashboardMCPServer:
    """MCP Server that exposes WandB dashboard management functionality."""

    def __init__(self, server_name: str = "wandb-mcp", version: str = "0.1.0"):
        self.app = Server(server_name)
        self.version = version
        self.config = WandBMCPConfig()

        # Try to initialize tools, but don't fail if wandb auth fails
        try:
            self.tools = WandBDashboardTools()
            self.authenticated = True
        except Exception as e:
            logger.warning(f"WandB authentication failed: {e}. Tools will have limited functionality.")
            self.tools = WandBDashboardToolsStub()  # Use a stub implementation instead of None
            self.authenticated = False

        self._setup_tools()
        self._setup_resources()

    def _setup_tools(self) -> None:
        """Register all WandB dashboard tools with the MCP server."""

        @self.app.list_tools()
        async def list_tools() -> List[types.Tool]:
            """List available WandB dashboard tools."""
            return [
                types.Tool(
                    name="create_dashboard",
                    description="Create a new WandB workspace dashboard with specified configuration",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of the dashboard"},
                            "entity": {
                                "type": "string",
                                "description": "WandB entity (user/team) that will own the dashboard",
                            },
                            "project": {
                                "type": "string",
                                "description": "WandB project the dashboard is associated with",
                            },
                            "description": {
                                "type": "string",
                                "description": "Optional description of the dashboard",
                                "default": "",
                            },
                            "sections": {
                                "type": "array",
                                "description": "List of dashboard sections with panels",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "panels": {"type": "array", "items": {"type": "object"}},
                                    },
                                },
                            },
                        },
                        "required": ["name", "entity", "project"],
                    },
                ),
                types.Tool(
                    name="update_dashboard",
                    description="Update an existing WandB dashboard",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "dashboard_url": {"type": "string", "description": "URL of the dashboard to update"},
                            "modifications": {
                                "type": "object",
                                "description": "Modifications to apply to the dashboard",
                            },
                        },
                        "required": ["dashboard_url", "modifications"],
                    },
                ),
                types.Tool(
                    name="list_dashboards",
                    description="List available dashboards for an entity/project",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "WandB entity to list dashboards for"},
                            "project": {
                                "type": "string",
                                "description": "WandB project to list dashboards for",
                                "default": None,
                            },
                            "filters": {"type": "object", "description": "Optional filters to apply", "default": {}},
                        },
                        "required": ["entity"],
                    },
                ),
                types.Tool(
                    name="get_dashboard_config",
                    description="Get the configuration of an existing dashboard",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "dashboard_url": {"type": "string", "description": "URL of the dashboard to inspect"}
                        },
                        "required": ["dashboard_url"],
                    },
                ),
                types.Tool(
                    name="add_panel",
                    description="Add a panel to an existing dashboard",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "dashboard_url": {"type": "string", "description": "URL of the dashboard"},
                            "section_name": {
                                "type": "string",
                                "description": "Name of the section to add the panel to",
                            },
                            "panel_type": {
                                "type": "string",
                                "enum": ["line_plot", "bar_plot", "scalar_chart", "scatter_plot"],
                                "description": "Type of panel to add",
                            },
                            "panel_config": {"type": "object", "description": "Configuration for the panel"},
                        },
                        "required": ["dashboard_url", "section_name", "panel_type", "panel_config"],
                    },
                ),
                types.Tool(
                    name="update_panel",
                    description="Update content of an existing panel in a dashboard",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "dashboard_url": {"type": "string", "description": "URL of the dashboard"},
                            "panel_identifier": {
                                "type": "object",
                                "description": "How to identify the panel to update",
                                "properties": {
                                    "index": {"type": "integer", "description": "Index of the panel (0-based)"},
                                    "search_text": {
                                        "type": "string",
                                        "description": "Search for panel containing this text",
                                    },
                                    "block_type": {
                                        "type": "string",
                                        "enum": ["markdown", "h1", "h2", "h3"],
                                        "description": "Type of block to find",
                                    },
                                    "occurrence": {
                                        "type": "integer",
                                        "description": "Which occurrence of block_type (0-based)",
                                        "default": 0,
                                    },
                                },
                                "oneOf": [
                                    {"required": ["index"]},
                                    {"required": ["search_text"]},
                                    {"required": ["block_type"]},
                                ],
                            },
                            "new_content": {"type": "string", "description": "New content for the panel"},
                        },
                        "required": ["dashboard_url", "panel_identifier", "new_content"],
                    },
                ),
                types.Tool(
                    name="remove_panel",
                    description="Remove an existing panel from a dashboard",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "dashboard_url": {"type": "string", "description": "URL of the dashboard"},
                            "panel_identifier": {
                                "type": "object",
                                "description": "How to identify the panel to remove",
                                "properties": {
                                    "index": {"type": "integer", "description": "Index of the panel (0-based)"},
                                    "search_text": {
                                        "type": "string",
                                        "description": "Search for panel containing this text",
                                    },
                                    "block_type": {
                                        "type": "string",
                                        "enum": ["markdown", "h1", "h2", "h3"],
                                        "description": "Type of block to find",
                                    },
                                    "occurrence": {
                                        "type": "integer",
                                        "description": "Which occurrence of block_type (0-based)",
                                        "default": 0,
                                    },
                                },
                                "oneOf": [
                                    {"required": ["index"]},
                                    {"required": ["search_text"]},
                                    {"required": ["block_type"]},
                                ],
                            },
                        },
                        "required": ["dashboard_url", "panel_identifier"],
                    },
                ),
                types.Tool(
                    name="create_custom_chart",
                    description="Create a custom chart/visualization with specified metrics and configuration",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "WandB entity"},
                            "project": {"type": "string", "description": "WandB project"},
                            "metrics": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of metrics to visualize",
                            },
                            "chart_type": {
                                "type": "string",
                                "enum": ["line_plot", "bar_plot", "scatter_plot", "custom_chart"],
                                "description": "Type of chart to create",
                            },
                            "config": {
                                "type": "object",
                                "description": "Chart configuration options",
                                "properties": {
                                    "title": {"type": "string", "description": "Chart title"},
                                    "x_label": {"type": "string", "description": "X-axis label"},
                                    "y_label": {"type": "string", "description": "Y-axis label"},
                                    "x_axis": {"type": "string", "description": "X-axis metric (default: 'step')"},
                                    "log_x": {"type": "boolean", "description": "Use logarithmic X-axis"},
                                    "log_y": {"type": "boolean", "description": "Use logarithmic Y-axis"},
                                    "smoothing": {"type": "number", "description": "Smoothing factor (0.0-1.0)"},
                                    "max_runs": {"type": "integer", "description": "Maximum runs to show"},
                                    "orientation": {
                                        "type": "string",
                                        "enum": ["vertical", "horizontal"],
                                        "description": "Bar chart orientation",
                                    },
                                    "show_regression": {
                                        "type": "boolean",
                                        "description": "Show regression line (scatter plots)",
                                    },
                                    "report_title": {"type": "string", "description": "Dashboard title"},
                                    "report_description": {"type": "string", "description": "Dashboard description"},
                                },
                            },
                        },
                        "required": ["entity", "project", "metrics", "chart_type", "config"],
                    },
                ),
                types.Tool(
                    name="list_available_metrics",
                    description="List available metrics for a project to use in dashboards",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "WandB entity"},
                            "project": {"type": "string", "description": "WandB project"},
                            "run_filters": {
                                "type": "object",
                                "description": "Optional filters for runs to analyze",
                                "default": {},
                            },
                        },
                        "required": ["entity", "project"],
                    },
                ),
                types.Tool(
                    name="clone_dashboard",
                    description="Clone an existing dashboard to create a new one",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source_url": {"type": "string", "description": "URL of the source dashboard to clone"},
                            "new_name": {"type": "string", "description": "Name for the cloned dashboard"},
                        },
                        "required": ["source_url", "new_name"],
                    },
                ),
                types.Tool(
                    name="bulk_delete_dashboards",
                    description="Bulk delete multiple WandB dashboards with confirmation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "dashboard_urls": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of dashboard URLs to delete",
                            },
                            "confirmed": {
                                "type": "boolean",
                                "description": "Confirmation flag - set to true after reviewing deletion details",
                                "default": False,
                            },
                        },
                        "required": ["dashboard_urls"],
                    },
                ),
            ]

        @self.app.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls by dispatching to appropriate methods."""
            try:
                if name == "create_dashboard":
                    result = await self.tools.create_dashboard(
                        name=arguments["name"],
                        entity=arguments["entity"],
                        project=arguments["project"],
                        description=arguments.get("description", ""),
                        sections=arguments.get("sections", []),
                    )
                elif name == "update_dashboard":
                    result = await self.tools.update_dashboard(
                        dashboard_url=arguments["dashboard_url"], modifications=arguments["modifications"]
                    )
                elif name == "list_dashboards":
                    result = await self.tools.list_dashboards(
                        entity=arguments["entity"],
                        project=arguments.get("project"),
                        filters=arguments.get("filters", {}),
                    )
                elif name == "get_dashboard_config":
                    result = await self.tools.get_dashboard_config(dashboard_url=arguments["dashboard_url"])
                elif name == "add_panel":
                    result = await self.tools.add_panel(
                        dashboard_url=arguments["dashboard_url"],
                        section_name=arguments["section_name"],
                        panel_type=arguments["panel_type"],
                        panel_config=arguments["panel_config"],
                    )
                elif name == "update_panel":
                    result = await self.tools.update_panel(
                        dashboard_url=arguments["dashboard_url"],
                        panel_identifier=arguments["panel_identifier"],
                        new_content=arguments["new_content"],
                    )
                elif name == "remove_panel":
                    result = await self.tools.remove_panel(
                        dashboard_url=arguments["dashboard_url"],
                        panel_identifier=arguments["panel_identifier"],
                    )
                elif name == "create_custom_chart":
                    result = await self.tools.create_custom_chart(
                        entity=arguments["entity"],
                        project=arguments["project"],
                        metrics=arguments["metrics"],
                        chart_type=arguments["chart_type"],
                        config=arguments["config"],
                    )
                elif name == "list_available_metrics":
                    result = await self.tools.list_available_metrics(
                        entity=arguments["entity"],
                        project=arguments["project"],
                        run_filters=arguments.get("run_filters", {}),
                    )
                elif name == "clone_dashboard":
                    result = await self.tools.clone_dashboard(
                        source_url=arguments["source_url"], new_name=arguments["new_name"]
                    )
                elif name == "bulk_delete_dashboards":
                    result = await self.tools.bulk_delete_dashboards(
                        dashboard_urls=arguments["dashboard_urls"],
                        confirmed=arguments.get("confirmed", False),
                    )
                else:
                    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

                return [types.TextContent(type="text", text=result)]

            except Exception as e:
                logger.error(f"Error calling tool {name}: {e}")
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    def _setup_resources(self) -> None:
        """Set up MCP resources for the server."""

        @self.app.list_resources()
        async def list_resources() -> List[types.Resource]:
            """List available resources."""
            return [
                types.Resource(
                    uri="wandb://dashboards",
                    name="WandB Dashboards",
                    description="Access to WandB workspace dashboards",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="wandb://metrics",
                    name="WandB Metrics",
                    description="Available metrics from WandB projects",
                    mimeType="application/json",
                ),
            ]

        @self.app.read_resource()
        async def read_resource(uri: str) -> str:
            """Read resource content."""
            if uri == "wandb://dashboards":
                # Return list of available dashboards
                return await self.tools.get_available_dashboards()
            elif uri == "wandb://metrics":
                # Return available metrics information
                return await self.tools.get_available_metrics_summary()
            else:
                raise ValueError(f"Unknown resource URI: {uri}")


async def main():
    """Main entry point for the WandB Dashboard MCP server."""

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info("Starting WandB Dashboard MCP Server...")

    try:
        server = WandBDashboardMCPServer()

        async with stdio_server() as (read_stream, write_stream):
            await server.app.run(read_stream, write_stream, server.app.create_initialization_options())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)


def cli_main():
    """CLI entry point for the server."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
