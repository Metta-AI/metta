"""MCP server for Metta run.py tool execution and discovery."""

from .server import RunToolMCPServer, cli_main

__all__ = ["RunToolMCPServer", "cli_main"]
__version__ = "0.1.0"


