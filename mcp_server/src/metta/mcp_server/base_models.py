"""
Base Pydantic models for common MCP server responses.

This module contains generic response models that are used across
multiple MCP server functions for consistent API responses.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenericSuccess(BaseModel):
    """Generic successful response."""

    success: bool = Field(default=True, description="Operation completed successfully")
    data: dict[str, str | int | float | bool] = Field(description="Response data")


class GenericError(BaseModel):
    """Generic error response."""

    success: bool = Field(default=False, description="Operation failed")
    error: str = Field(description="Error message describing what went wrong")


class GenericOperationResult(BaseModel):
    """Generic operation result."""

    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Result message or confirmation")
    data: dict[str, str | int | float | bool | None] = Field(description="Additional result data")


class SearchResult(BaseModel):
    """Search result with pagination."""

    results: list[dict[str, str | int | float | bool | None]] = Field(description="List of search results")
    total: int = Field(description="Total number of results available")
    limit: int = Field(description="Number of results per page")
    offset: int = Field(description="Offset into the result set")


class UserInfo(BaseModel):
    """Current user information."""

    email: str = Field(description="User's email address")
    authenticated: bool = Field(description="Whether user is properly authenticated")


class TokenCreationResult(BaseModel):
    """Result of token creation."""

    token_id: str = Field(description="Unique identifier for the created token")
    token_value: str = Field(description="The actual token string (only shown once)")
    name: str = Field(description="Human-readable name for the token")
    created_at: str = Field(description="ISO timestamp when token was created")


class CliTokenResult(BaseModel):
    """CLI token creation with redirect."""

    redirect_url: str = Field(description="URL to redirect to with token parameter")
    token_id: str = Field(description="Unique identifier for the created token")
