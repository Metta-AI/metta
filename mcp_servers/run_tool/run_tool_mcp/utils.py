"""Utility functions for the Run Tool MCP server."""

import json
from typing import Any


def format_command_preview(tool_path: str, arguments: dict[str, Any] | None = None) -> str:
    """Format a command preview string from tool path and arguments dict."""
    cmd_parts = ["./tools/run.py", tool_path]

    if arguments:
        for key, value in arguments.items():
            if value is None:
                continue
            if isinstance(value, bool):
                value_str = str(value).lower()
            elif isinstance(value, (list, dict)):
                value_str = json.dumps(value)
            else:
                value_str = str(value)
            cmd_parts.append(f"{key}={value_str}")

    return " ".join(cmd_parts)


def normalize_arguments(arguments: Any) -> dict[str, Any] | None:
    """Normalize arguments to a dict, handling JSON strings."""
    if arguments is None:
        return None
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return None


def determine_error_type(exit_code: int, stderr: str) -> str | None:
    """Determine error type from exit code and stderr content."""
    if exit_code == 0:
        return None
    if exit_code == 130:
        return "interrupted"
    if "Unknown arguments" in stderr:
        return "unknown_arguments"
    if "Error creating tool configuration" in stderr:
        return "tool_construction_error"
    if "Error applying override" in stderr:
        return "override_error"
    if "Tool invocation failed" in stderr:
        return "invocation_error"
    return "execution_error"

