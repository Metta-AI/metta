"""
MCP server implementation for codebot.

This module implements a Model Context Protocol (MCP) server that exposes
codebot's AI-powered development assistance tools to MCP clients.
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

from .commands import run_summarize_command

logger = logging.getLogger(__name__)


class CodebotMCPServer:
    """MCP Server that exposes codebot functionality as MCP tools."""

    def __init__(self, server_name: str = "codebot-mcp-server", version: str = "1.0.0"):
        self.app = Server(server_name)
        self.version = version
        self._setup_tools()
        self._setup_resources()

    def _setup_tools(self) -> None:
        """Register all codebot tools with the MCP server."""

        @self.app.list_tools()
        async def list_tools() -> List[types.Tool]:
            """List available codebot tools."""
            return [
                types.Tool(
                    name="summarize",
                    description="Generate an AI-powered summary of code files",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of file or directory paths to analyze. "
                                "Defaults to current directory if not provided.",
                            },
                            "token_limit": {
                                "type": "integer",
                                "default": 2000,
                                "description": "Maximum tokens for the summary (default: 2000)",
                            },
                            "no_cache": {
                                "type": "boolean",
                                "default": False,
                                "description": "Bypass cache and generate fresh summary",
                            },
                        },
                    },
                ),
                types.Tool(
                    name="context",
                    description="Show the context that would be sent to AI commands",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of file or directory paths to analyze. "
                                "Defaults to current directory if not provided.",
                            }
                        },
                    },
                ),
            ]

        @self.app.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls."""
            if name == "summarize":
                return await self._handle_summarize_tool(arguments)
            elif name == "context":
                return await self._handle_context_tool(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _handle_summarize_tool(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle summarize tool calls."""
        paths = arguments.get("paths")
        token_limit = arguments.get("token_limit", 2000)
        no_cache = arguments.get("no_cache", False)

        try:
            # Use current directory if no paths provided
            if paths is None or len(paths) == 0:
                paths = ["."]

            # Run the summarize command
            result = await run_summarize_command(
                paths=paths, token_limit=token_limit, working_dir=Path.cwd(), no_cache=no_cache
            )

            if not result.success:
                return [types.TextContent(type="text", text=f"Error: {result.summary}")]

            # Format the response
            response_text = f"✓ {result.summary}"

            # Add metadata if available
            if result.metadata:
                meta = result.metadata
                details = []
                if "input_token_count" in meta:
                    details.append(f"Input: ~{meta['input_token_count']:,} tokens")
                if "output_file" in meta:
                    details.append(f"Output: {meta['output_file']}")
                if "cached" in meta and meta["cached"]:
                    details.append(f"Cached: ✓ (key: {meta['cache_key'][:8]}...)")
                elif "cached" in meta and not meta["cached"]:
                    details.append(f"Fresh: ✓ (key: {meta['cache_key'][:8]}...)")

                if details:
                    response_text += "\n" + "\n".join(f"  {detail}" for detail in details)

            # Apply changes and show what files were created
            modified_files = result.apply_changes()
            if modified_files:
                response_text += f"\n  Written: {', '.join(modified_files)}"

            return [types.TextContent(type="text", text=response_text)]

        except Exception as e:
            logger.error(f"Error in summarize tool: {e}")
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    async def _handle_context_tool(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle context tool calls."""
        paths = arguments.get("paths")

        try:
            # Import here to avoid circular imports
            current_dir = os.path.dirname(__file__)
            codeclip_path = os.path.join(current_dir, "..", "codeclip")

            if codeclip_path not in sys.path:
                sys.path.insert(0, codeclip_path)

            from file import get_context

            path_list = paths if paths else ["."]

            content, token_info = get_context(
                paths=path_list,
                extensions=(".py", ".ts", ".tsx", ".js", ".java", ".cpp", ".h"),
                include_git_diff=False,
                readmes_only=False,
            )

            # Show token info
            total_tokens = token_info.get("total_tokens", 0)
            total_files = token_info.get("total_files", 0)

            summary = "Context Summary:\n"
            summary += f"  Files: {total_files}\n"
            summary += f"  Tokens: ~{total_tokens:,}\n\n"

            # Include the content
            full_response = summary + content

            return [types.TextContent(type="text", text=full_response)]

        except ImportError as e:
            error_msg = f"Error: Cannot import codeclip: {e}"
            return [types.TextContent(type="text", text=error_msg)]
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return [types.TextContent(type="text", text=error_msg)]

    def _setup_resources(self) -> None:
        """Register MCP resources."""

        @self.app.list_resources()
        async def list_resources() -> List[types.Resource]:
            """List available codebot resources."""
            return [
                types.Resource(
                    uri="codebot://summaries",
                    name="Code Summaries",
                    description="AI-generated code summaries and analysis results",
                    mimeType="text/markdown",
                ),
                types.Resource(
                    uri="codebot://context",
                    name="Code Context",
                    description="Context information for AI commands",
                    mimeType="text/plain",
                ),
            ]

        @self.app.read_resource()
        async def read_resource(uri: str) -> str:
            """Read a codebot resource."""
            if uri == "codebot://summaries":
                # Look for recent summaries in .codebot/summaries
                summaries_dir = Path(".codebot/summaries")
                if summaries_dir.exists():
                    summary_files = list(summaries_dir.glob("*.md"))
                    if summary_files:
                        # Return the most recent summary
                        latest_summary = max(summary_files, key=lambda p: p.stat().st_mtime)
                        return latest_summary.read_text()
                return "No summaries found. Run the 'summarize' tool first."

            elif uri == "codebot://context":
                return "Use the 'context' tool to view current code context information."

            else:
                raise ValueError(f"Unknown resource URI: {uri}")

    async def run(self) -> None:
        """Run the MCP server."""
        logger.info(f"Starting Codebot MCP server (version {self.version})")

        async with stdio_server() as streams:
            await self.app.run(streams[0], streams[1], self.app.create_initialization_options())


async def main() -> None:
    """Main entry point for the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,  # MCP uses stdout for protocol messages
    )

    server = CodebotMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
