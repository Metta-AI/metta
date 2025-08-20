"""
Database-related Pydantic models for MCP server.

This module contains models for SQL queries, table metadata, and
database operations used in the MCP server's database functionality.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TableMetadata(BaseModel):
    """Database table metadata."""

    name: str = Field(description="Table name")
    column_count: int = Field(description="Number of columns in the table")
    row_count: int = Field(description="Number of rows in the table")


class TableList(BaseModel):
    """List of database tables."""

    tables: list[TableMetadata] = Field(description="List of table metadata objects")


class SQLQueryResult(BaseModel):
    """SQL query execution result."""

    rows: list[dict[str, str | int | float | bool | None]] = Field(description="Query result rows")
    columns: list[str] = Field(description="Column names in the result")
    row_count: int = Field(description="Number of rows returned")
    execution_time_ms: float = Field(default=0.0, description="Query execution time in milliseconds")


class SQLQueryError(BaseModel):
    """SQL query execution error."""

    error: str = Field(description="Error message describing why the query failed")
    sql_statement: str = Field(description="The SQL statement that caused the error")


class AIQueryResult(BaseModel):
    """AI-generated SQL query result."""

    generated_query: str = Field(description="AI-generated SQL query")
    explanation: str = Field(description="Explanation of what the query does")
    confidence: float = Field(description="AI confidence in the generated query (0.0-1.0)")


class ValidationResult(BaseModel):
    """Configuration validation result with detailed info."""

    valid: bool = Field(description="Whether the configuration is valid")
    config_data: dict[str, Any] | list[Any] | None = Field(description="Parsed configuration data if valid")
    errors: list[str] = Field(description="List of validation errors if invalid")
    config_path: str = Field(description="Path to the configuration file validated")
