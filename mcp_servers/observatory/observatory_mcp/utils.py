"""Observatory MCP Server Utilities."""

import logging
from typing import Any

from observatory_mcp.models import ErrorResponse, SuccessResponse

logger = logging.getLogger(__name__)


def format_success_response(data: Any) -> str:
    """Format success response as JSON string."""
    return SuccessResponse(data=data).model_dump_json(indent=2, exclude_none=True)


def format_error_response(error: Exception, tool_name: str, context: str | None = None) -> str:
    """Format error as JSON string for MCP response."""
    return ErrorResponse(
        tool=tool_name, message=str(error), error_type=type(error).__name__, context=context
    ).model_dump_json(indent=2, exclude_none=True)


def handle_backend_error(error: Exception, tool_name: str) -> str:
    """Handle errors from backend API calls."""
    logger.error(f"Backend error in {tool_name}: {error}", exc_info=True)

    error_type = type(error).__name__
    error_message = str(error)

    if "Connection" in error_type or "ConnectTimeout" in error_type:
        return format_error_response(
            Exception("Backend connection failed. Is the backend running at the configured URL?"),
            tool_name,
            context="Check METTA_MCP_BACKEND_URL environment variable",
        )

    if "HTTPStatusError" in error_type or "HTTPError" in error_type:
        if "401" in error_message or "403" in error_message:
            return format_error_response(
                Exception("Authentication failed. Check your machine token."),
                tool_name,
                context="Set METTA_MCP_MACHINE_TOKEN environment variable",
            )
        if "404" in error_message:
            return format_error_response(
                Exception("Backend endpoint not found. The backend may be outdated."), tool_name
            )
        if "500" in error_message:
            return format_error_response(Exception("Backend server error. Check backend logs."), tool_name)

    return format_error_response(error, tool_name)


def handle_validation_error(error: Exception, tool_name: str, field: str | None = None) -> str:
    """Handle validation errors (missing required fields, invalid types, etc.)."""
    error_msg = str(error)
    if field:
        error_msg = f"Validation error for field '{field}': {error_msg}"

    return format_error_response(
        ValueError(error_msg), tool_name, context="Check tool arguments and ensure all required fields are provided"
    )


def serialize_response_data(data: Any) -> Any:
    """Serialize response data to make it JSON-compatible."""
    if hasattr(data, "model_dump"):
        return data.model_dump(mode="json")
    if isinstance(data, dict):
        return {key: serialize_response_data(value) for key, value in data.items()}
    if isinstance(data, list):
        return [serialize_response_data(item) for item in data]
    return data
